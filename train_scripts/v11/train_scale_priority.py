"""Fine-tune the fused RLNN encoder on independent-set teacher ensembles."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"models"))
from scale_priority import PriorityScorer


REGIMES = ("RB_small", "RB_large", "ER_small")


def greedy(order, row, col):
    available = np.ones(len(row)-1, dtype=bool)
    size = 0
    for i in order:
        if available[i]:
            available[i] = False
            available[col[row[i]:row[i+1]]] = False
            size += 1
    return size


def examples(model, directory, device):
    rows = []
    for entry in json.loads((directory/"tensor_manifest.json").read_text())["graphs"]:
        path = directory/entry["tensor"]
        values = torch.load(path, weights_only=True)
        n = values["num_nodes"]
        u, v = values["edge_index"]
        row = torch.zeros(n+1, dtype=torch.long)
        torch.cumsum(torch.bincount(u, minlength=n), dim=0, out=row[1:])
        norm, structure = model.prepare(row.to(device), v.to(device))
        masks = np.load(path.with_suffix(".targets.npz"))["masks"]
        sizes = masks.sum(axis=1)
        largest = np.argsort(-sizes.astype(np.int64), kind="stable")[:4]
        probabilities = torch.from_numpy(masks[largest].mean(axis=0).astype(np.float32)).to(device)
        crow, col = row.numpy(), v.numpy()
        degree_size = greedy(np.argsort(np.diff(crow), kind="stable"), crow, col)
        rows.append((norm, structure, probabilities, int(sizes.max()), crow, col, degree_size))
    return rows


def teacher_loss(scores, target):
    return -0.5*((target * F.logsigmoid(scores)).sum()/target.sum() +
                 ((1 - target) * F.logsigmoid(-scores)).sum()/(1 - target).sum())


def degree_matched_pairs(example, rng):
    target = example[2].cpu().numpy()
    degree = np.diff(example[4])
    pos, neg = [], []
    for i in np.flatnonzero(target > 0):
        candidates = np.flatnonzero((abs(degree-degree[i]) <= 2) & (target < target[i]))
        if len(candidates):
            sampled = rng.choice(candidates, size=12, replace=True)
            pos.extend([i]*len(sampled))
            neg.extend(sampled)
    return torch.tensor([pos, neg], dtype=torch.long, device=example[2].device)


@torch.inference_mode()
def evaluate(model, groups):
    result = {}
    for regime, records in groups.items():
        scores, losses, teacher_sizes, degree_sizes = [], [], [], []
        for norm, structure, target, teacher, row, col, degree in records:
            logits = model(norm, structure)
            order = np.argsort(-logits.cpu().numpy(), kind="stable")
            scores.append(greedy(order, row, col))
            losses.append(float(teacher_loss(logits, target)))
            teacher_sizes.append(teacher)
            degree_sizes.append(degree)
        result[regime] = dict(size=float(np.mean(scores)), loss=float(np.mean(losses)),
            teacher=float(np.mean(teacher_sizes)), degree=float(np.mean(degree_sizes)),
            normalized_gain=float(np.mean((np.array(scores)-degree_sizes)/np.array(teacher_sizes))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=8707)
    parser.add_argument("--variant", choices=("fused_readout","degree_residual"), default="fused_readout")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = PriorityScorer.from_fused(args.fused, device, args.variant)
    train = [item for regime in REGIMES for item in examples(model, args.data/regime/"train", device)]
    valid = {regime: examples(model, args.data/regime/"validation", device) for regime in REGIMES}
    use_pairs = args.variant == "degree_residual"
    pairs = [degree_matched_pairs(item, rng) for item in train] if use_pairs else []
    optimizer = torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=0.00008),
                                   dict(params=model.priority_head.parameters(),lr=0.0015)], weight_decay=0.001)
    config = dict(architecture="Fused RLNN residual GCN 1-128-5-1 + linear priority residual",
        initial_checkpoint=str(args.fused), graph_data=str(args.data), epochs=args.epochs, seed=args.seed,
        loss="graph-balanced soft-label BCE on top four of eight OnlineMIS feasible sets",
        encoder_lr=0.00008, head_lr=0.0015, weight_decay=0.001, gradient_clip=2.0,
        accumulate_graphs=4, validation="deterministic greedy by predicted priority",
        selection="largest graph-mean greedy gain over degree / best teacher size, ties by smallest validation loss; epochs >=1",
        device=torch.cuda.get_device_name(device), regimes=REGIMES, torch=str(torch.__version__),
        train_graphs=len(train), validation_graphs={regime:len(records) for regime,records in valid.items()})
    if use_pairs:
        config.update(architecture="Fused RLNN GCN + standardized degree skip + bounded 24-wide residual MLP",
            loss="0.7 balanced BCE + 0.3 teacher-frequency-weighted pairwise logistic loss",
            pair_selection="same graph, degree difference <=2, 12 lower-frequency negatives per positive",
            residual_amplitude=0.35, selection="largest mean normalized gain over degree, ties by least validation loss, epoch zero eligible")
    (output/"config.json").write_text(json.dumps(config, indent=2)+"\n")
    best = {}
    with (output/"epochs.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch","regime","train_loss","size","loss","teacher","degree","normalized_gain"))
        writer.writeheader()
        for epoch in range(args.epochs+1):
            training_loss = 0.0
            if epoch:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for step, i in enumerate(rng.permutation(len(train)), 1):
                    norm, structure, target = train[i][:3]
                    scores = model(norm, structure)
                    loss = teacher_loss(scores, target)
                    if use_pairs:
                        pos, neg = pairs[i]
                        if len(pos):
                            weights = target[pos]-target[neg]
                            ranking_loss = (weights*F.softplus(scores[neg]-scores[pos])).sum()/weights.sum()
                            loss = 0.7*loss + 0.3*ranking_loss
                    (loss/4).backward()
                    training_loss += loss.detach()
                    if step % 4 == 0 or step == len(train):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                training_loss = float(training_loss / len(train))
            model.eval()
            metrics = evaluate(model, valid)
            for regime, data in metrics.items():
                writer.writerow(dict(epoch=epoch,regime=regime,train_loss=training_loss,**data))
            stream.flush()
            print(f"epoch {epoch:02d} loss {training_loss:.4f} sizes " +
                  " ".join(f"{key}={data['size']:.3f}" for key,data in metrics.items()), flush=True)
            if epoch or use_pairs:
                ranking = (float(np.mean([v["normalized_gain"] for v in metrics.values()])),
                           -float(np.mean([v["loss"] for v in metrics.values()])))
                if not best or ranking > tuple(best["ranking"]):
                    best = dict(epoch=epoch, ranking=ranking, validation=metrics)
                    torch.save(dict(model=model.state_dict(), epoch=epoch, regime="shared",variant=args.variant), output/"best.pt")
                    (output/"selection.json").write_text(json.dumps(best, indent=2)+"\n")
    print("All selection decisions frozen.",flush=True)


if __name__ == "__main__":
    main()
