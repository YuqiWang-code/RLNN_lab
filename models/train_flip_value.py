"""Fit a flip-value head on paired continuations of a frozen RLNN policy."""
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from eval_v1 import ROOT
from main import build_parser, set_seed
from mis_v1 import FlipValue, adjacency
from models import edge2adj
from problems.mis import Dataset
from solvers import RLNN


def uniforms_like(tensor, generator):
    return torch.rand(tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator)


@torch.no_grad()
def paired_examples(solver, graph, horizon, pairs, generator):
    adj = adjacency(graph)
    normalized = edge2adj(graph.edge_index, graph.num_nodes)
    features = []
    targets = []
    probes = {solver.num_t * i // 5 for i in range(1, 5)}
    for step, x in enumerate(solver.sample_states(graph, adj, generator=generator, normalized=normalized)):
        if step not in probes:
            continue
        states = x[:, :pairs]
        logits = solver.flip_logits(states, graph, adj, normalized)
        columns = torch.arange(pairs, device=x.device)
        nodes = torch.randint(graph.num_nodes, (pairs,), device=x.device, generator=generator)
        features.append(FlipValue.features(states, logits, adj, solver.problem.beta)[nodes, columns])
        shared = uniforms_like(states, generator)
        proposed = torch.where(shared < logits.sigmoid(), 1 - states, states)
        on, off = proposed.clone(), proposed.clone()
        on[nodes, columns] = 1 - states[nodes, columns]
        off[nodes, columns] = states[nodes, columns]
        branch = torch.cat((on, off), dim=1)
        for _ in range(horizon):
            prob = solver.flip_logits(branch, graph, adj, normalized).sigmoid()
            shared = uniforms_like(states, generator).repeat(1, 2)
            branch = torch.where(shared < prob, 1 - branch, branch)
        size = -solver.problem.decode_gradient_flip(branch, adj, graph.b).float()
        targets.append(size[:pairs] - size[pairs:])
        if step == max(probes):
            break
    return torch.cat(features), torch.cat(targets)


def collect(solver, folder, count, args, seed):
    data = Dataset(folder, solver.dtype, count)
    device = solver.accelerator.device
    features = []
    targets = []
    for index, graph in enumerate(DataLoader(data, batch_size=None, num_workers=args.num_workers, pin_memory=True)):
        generator = torch.Generator(device=device).manual_seed(seed + 2 * index)
        x, y = paired_examples(solver, graph.to(device), args.horizon, args.pairs, generator)
        features.append(x.cpu())
        targets.append(y.cpu())
        if (index + 1) % 100 == 0:
            print(f"{folder}: {index + 1}/{len(data)} graphs", flush=True)
    return torch.cat(features).to(device), torch.cat(targets).to(device)


def main():
    parser = build_parser()
    parser.set_defaults(method="RLNN", problem="mis", num_t=100, num_k=20, num_h=128,
                        num_l=5, mixed_precision="fp16", num_workers=4,
                        train_path=str(ROOT / "models/data/MIS_tensors/RB_small_train"),
                        valid_path=str(ROOT / "models/data/MIS_tensors/RB_small_val"),
                        test_path=str(ROOT / "models/data/author_rld4co/MIS/RB_small_test"),
                        train_samples=1000, valid_samples=100, lr=0.001, epochs=20)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--result_dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.pairs <= args.num_k or args.num_t < 5:
        parser.error("require 1 <= pairs <= num_k and at least five sampling steps")
    if args.do_train:
        parser.error("the RLNN policy is loaded frozen from --save_dir")
    set_seed(args.seed)
    solver = RLNN(args)
    solver.prepare()
    solver.model.eval()
    device = solver.accelerator.device
    result = args.result_dir
    result.mkdir(parents=True, exist_ok=True)
    checkpoint = result / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    train_x, train_y = collect(solver, args.train_path, args.train_samples, args, args.seed)
    val_x, val_y = collect(solver, args.valid_path, args.valid_samples, args, args.seed + 1)
    generation_seconds = perf_counter() - started

    head = FlipValue().to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        for selection in torch.randperm(len(train_y), device=device).split(1024):
            loss = F.smooth_l1_loss(head(train_x[selection]).squeeze(-1), train_y[selection])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            pred = head(val_x).squeeze(-1)
            mae = (pred - val_y).abs().mean().item()
        history.append({"epoch": epoch, "val_mae": mae})
        if mae < best:
            best = mae
            torch.save(head.state_dict(), checkpoint / "best.pt")
        print(f"Epoch {epoch}: validation MAE {mae:.4f}", flush=True)

    metadata = dict(version="v1", mode="value_train", status="completed", run=result.name,
                    config=vars(args), started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                    wall_time_s=perf_counter() - started, generation_time_s=generation_seconds,
                    train_pairs=len(train_y), val_pairs=len(val_y), val_mae=best,
                    val_zero_mae=val_y.abs().mean().item(), val_delta=val_y.mean().item(),
                    val_positive_rate=(val_y > 0).float().mean().item(), history=history)
    (result / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
