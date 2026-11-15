"""Learn to rank feasible exchanges by their paired continuation quality."""
import csv
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from scipy.stats import rankdata
from torch.nn import functional as F

from eval_v2 import TRAIN, dataset_at, load_policy, policy_parser, search, write_run
from mis_v2_exchange import ExchangeGraph, action_values
from mis_v2_value import PAVE


@torch.no_grad()
def examples(args, solver, path, count, split):
    data = dataset_at(path, solver.dtype, count)
    states = []
    for index, graph in enumerate(data):
        topology = ExchangeGraph(graph.num_nodes, graph.edge_index)
        selected, _ = search(solver, graph.to(solver.accelerator.device), topology, args.seed + index)
        actions = topology.actions(selected, cap=args.candidate_cap, larger=args.larger)
        if actions:
            quality = action_values(topology, selected, actions, cap=args.candidate_cap,
                                    rounds=args.exchange_rounds, larger=args.larger)
            features = [topology.features(selected, action) for action in actions]
            heuristic = min(range(len(actions)), key=lambda i: topology.heuristic(selected, actions[i]))
            states.append((torch.tensor(features), torch.tensor(quality), heuristic))
        if (index + 1) % 50 == 0:
            print(f"{split} {index + 1}/{len(data)} graphs, {len(states)} with actions", flush=True)
    return states


def pairs(states):
    better, worse = [], []
    for features, quality, _ in states:
        first, second = torch.where(quality.unsqueeze(1) > quality.unsqueeze(0))
        better.append(features[first])
        worse.append(features[second])
    return torch.cat(better), torch.cat(worse)


@torch.no_grad()
def validate(model, states):
    rows = []
    for features, quality, heuristic in states:
        upper = quality.max()
        if quality.min() == upper:
            continue
        score = model(features)
        first, second = torch.where(quality.unsqueeze(1) > quality.unsqueeze(0))
        order = score[first] - score[second]
        concordance = (order.gt(0).float() + 0.5 * order.eq(0)).mean()
        rank_score = torch.as_tensor(rankdata(score))
        rank_quality = torch.as_tensor(rankdata(quality))
        correlation = torch.corrcoef(torch.stack((rank_score, rank_quality)))[0, 1].nan_to_num()
        predicted = quality[int(score.argmax())]
        rows.append(dict(
            top1=float(predicted == upper), heuristic_top1=float(quality[heuristic] == upper),
            rule_top1=float(quality[0] == upper), random_top1=float((quality == upper).float().mean()),
            pair_accuracy=float(concordance), spearman=float(correlation),
            predicted_size=float(predicted), heuristic_size=float(quality[heuristic]),
        ))
    return {key: mean(row[key] for row in rows) for key in rows[0]}


def main():
    parser = policy_parser()
    parser.set_defaults(split="validation", train_path=str(TRAIN), train_samples=500, test_samples=100, epochs=30, lr=0.001)
    args = parser.parse_args()
    result = args.result_dir
    checkpoints = result / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    solver = load_policy(args)
    train = examples(args, solver, args.train_path, args.train_samples, "train")
    valid = examples(args, solver, args.test_path, args.test_samples, "validation")
    generation_time = perf_counter() - started
    train_informative = sum(quality.min() < quality.max() for _, quality, _ in train)
    val_informative = sum(quality.min() < quality.max() for _, quality, _ in valid)
    train_informative, val_informative = int(train_informative), int(val_informative)
    metadata = dict(version="v2", mode="value_train", run=result.name, config=vars(args),
                    train_seed=args.train_seed, train_states=len(train), val_states=len(valid),
                    train_informative=train_informative, val_informative=val_informative,
                    generation_time_s=generation_time, started_at=started_at)
    if not train_informative or not val_informative:
        write_run(result, dict(metadata, status="skipped", reason="no non-tied pairs in a split"))
        print("No non-tied comparisons in a split", flush=True)
        return
    better, worse = pairs(train)
    model = PAVE()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = torch.Generator().manual_seed(args.train_seed)
    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in torch.randperm(len(better), generator=rng).split(256):
            loss = F.softplus(model(worse[batch]) - model(better[batch])).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        metrics = dict(epoch=epoch, loss=mean(losses), **validate(model, valid))
        history.append(metrics)
        key = metrics["top1"], metrics["pair_accuracy"]
        if best is None or key > (best["top1"], best["pair_accuracy"]):
            best = metrics
            torch.save(model.state_dict(), checkpoints / "best.pt")
        print(f"Epoch {epoch}: loss {metrics['loss']:.4f}, top1 {metrics['top1']:.3f}, pair {metrics['pair_accuracy']:.3f}", flush=True)
    with (result / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    write_run(result, dict(metadata, status="completed", best=best, train_pairs=len(better),
                           finished_at=datetime.now(timezone.utc).isoformat(), wall_time_s=perf_counter() - started))


if __name__ == "__main__":
    main()
