"""Pure supervised fit of the one-shot static direction to solution-frequency labels.

No search objective, no rollouts: the node score is trained only to rank nodes by how
often they appear in a set of independently seeded high-quality solutions. The saved
checkpoint is the one with the best held-out AUC, and it loads through the same
``preflight_policy`` path as the search-trained adapters.
"""
import csv
import json
import math
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from eval_v2 import TRAIN, dataset_at, write_run
from main import build_parser, set_seed
from mis_v1 import adjacency
from mis_v7_context import V7StaticContextGCN
from v7_runs import backbone, digest, provenance
from v7_structure import offline_structure


def parse():
    parser = build_parser()
    parser.set_defaults(num_h=128, num_l=5, static_features=8, v7_width=32, v7_d_state=16,
                        v7_order_seed=7301, v7_order_views=1, static_node_limit=1., static_constant_limit=.25)
    parser.add_argument("--labels", type=Path, required=True, help="frequency labels from --train-oracle freq")
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--reference_seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--val_samples", type=int, default=100)
    args = parser.parse_args()
    args.context_mode = "static_centered"
    if args.static_features != 8 or args.v7_order_views != 1:
        parser.error("The supervised direction uses the 8 static features and one order view")
    if args.train_samples <= args.val_samples:
        parser.error("Use more training graphs than held-out graphs")
    if args.epochs < 1:
        parser.error("Use at least one epoch")
    return args


def build(args, device):
    model = V7StaticContextGCN(args.num_h, args.num_l, "static_centered", args.v7_width, args.v7_d_state, None,
                               static_node_limit=args.static_node_limit, static_encoder="mlp",
                               constant_limit=args.static_constant_limit, feature_dim=args.static_features,
                               bias_seed=0, oracle_path=None, order_seed=args.v7_order_seed,
                               views=args.v7_order_views).to(device)
    missing, extra = model.load_state_dict(
        torch.load(backbone(args.reference_seed), map_location="cpu", weights_only=True), strict=False)
    if extra or any(not key.startswith("adapter.") for key in missing):
        raise ValueError("The warm start must carry the frozen backbone only")
    model.requires_grad_(False)
    adapter = model.adapter
    with torch.no_grad():
        adapter.alpha.fill_(math.atanh(.5))
        adapter.constants.zero_()
    adapter.alpha.requires_grad_(False)
    adapter.constants.requires_grad_(False)
    adapter.context.requires_grad_(True)
    adapter.offline_features = True
    return model, adapter


def graph_loss(adapter, topology, freq, device):
    score = adapter.context(topology.features)[:, 0]
    target = torch.as_tensor(np.asarray(freq, dtype=np.float32), device=device)
    positive = target.sum().clamp_min(1.)
    weight = ((target.numel() - positive) / positive).clamp(1., 30.)
    return F.binary_cross_entropy_with_logits(score, target, pos_weight=weight), target.numel()


@torch.no_grad()
def evaluate(adapter, graphs, freqs, order, device):
    adapter.eval()
    scores, targets = [], []
    for index in order:
        graph = graphs[index]
        topology = adapter.topology(graph, adjacency(graph))
        scores.append(adapter.context(topology.features)[:, 0].float().cpu().numpy())
        targets.append(np.asarray(freqs[index], dtype=np.float32))
    labels = np.concatenate([(target >= .5).astype(np.int8) for target in targets])
    values = np.concatenate(scores)
    auc = float(roc_auc_score(labels, values)) if 0 < labels.sum() < labels.size else float("nan")
    rho = [float(spearmanr(score, target).statistic) for score, target in zip(scores, targets)]
    rho = [value for value in rho if np.isfinite(value)]
    adapter.train()
    return dict(val_auc=auc, val_spearman=float(np.mean(rho)) if rho else 0.,
                val_positive_rate=float(labels.mean()), val_graphs=len(order))


def main():
    args = parse()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, adapter = build(args, device)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    dataset = dataset_at(TRAIN, torch.float32, args.train_samples)
    graph_ids = [Path(path).stem for path in dataset.files]
    missing = [gid for gid in graph_ids if gid not in labels]
    if missing:
        raise ValueError(f"{args.labels} lacks {missing[0]}; regenerate with --train-oracle freq "
                         f"--samples {args.train_samples + args.val_samples}")
    graphs, freqs = [], []
    for graph, gid in zip(dataset, graph_ids):
        graph = graph.to(device)
        graphs.append(graph)
        freqs.append(labels[gid])
    train_order = list(range(len(graphs) - args.val_samples))
    val_order = list(range(len(graphs) - args.val_samples, len(graphs)))
    mean_values = torch.cat([torch.tensor(offline_structure(graphs[i].num_nodes,
                                                           graphs[i].edge_index.cpu().numpy().tobytes(),
                                                           args.v7_order_seed + graphs[i].idx,
                                                           None, args.v7_order_views)[0], dtype=torch.float32)
                             for i in train_order])
    adapter.feature_mean.copy_(mean_values.mean(0))
    adapter.feature_std.copy_(mean_values.std(0, correction=0).clamp_min(1e-4))
    normalization = dict(mean=adapter.feature_mean.tolist(), std=adapter.feature_std.tolist())
    norm_digest = sha256(adapter.feature_mean.detach().cpu().numpy().tobytes()
                         + adapter.feature_std.detach().cpu().numpy().tobytes()).hexdigest()

    args.result_dir.mkdir(parents=True, exist_ok=False)
    save_dir = args.result_dir / "checkpoints"
    save_dir.mkdir()
    optimizer = torch.optim.Adam(adapter.context.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    history, best = [], dict(val_auc=-math.inf, epoch=0)
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = rng.permutation(train_order)
        total, seen = 0.0, 0
        for start in range(0, len(permutation), args.graph_batch_size):
            batch = [int(index) for index in permutation[start:start + args.graph_batch_size]]
            optimizer.zero_grad()
            loss_sum = torch.zeros((), device=device)
            for index in batch:
                topology = adapter.topology(graphs[index], adjacency(graphs[index]))
                loss, nodes = graph_loss(adapter, topology, freqs[index], device)
                loss_sum = loss_sum + loss * nodes
                seen += nodes
            (loss_sum / seen).backward()
            optimizer.step()
            total += float(loss_sum.detach())
        metrics = evaluate(adapter, graphs, freqs, val_order, device)
        row = dict(epoch=epoch, train_loss=total / max(1, seen), **metrics,
                   seconds=perf_counter() - started)
        history.append(row)
        if metrics["val_auc"] > best["val_auc"]:
            best = dict(val_auc=metrics["val_auc"], epoch=epoch)
            torch.save(model.state_dict(), save_dir / "best.pt")
        print(json.dumps(row), flush=True)

    with (args.result_dir / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    record = dict(**provenance(args.reference_seed), version="static_direction", arm="supervised",
                  status="completed", config=vars(args), feature_normalization=normalization,
                  feature_norm_sha256=norm_digest, labels_sha256=digest(args.labels),
                  train_graphs=len(train_order), val_graphs=len(val_order), best=best,
                  params=sum(p.numel() for p in model.parameters()),
                  trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  train_wall_s=perf_counter() - started,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  finished_at=datetime.now(timezone.utc).isoformat(),
                  checkpoint_sha256=digest(save_dir / "best.pt"))
    write_run(args.result_dir, record)
    print(json.dumps(dict(best=best, checkpoint=str(save_dir / "best.pt")), indent=2))


if __name__ == "__main__":
    main()
