"""Train the upstream GCON layer on the shared RB graph split."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "reference_methods/GCON/source"
sys.path[:0] = [str(SOURCE), str(ROOT / "models")]

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from eval_v1 import TEST, add_evaluation_arguments, evaluate
from problems.mis import Dataset


def features(graph):
    neighbors = torch.zeros((graph.num_nodes, graph.num_nodes), device=graph.edge_index.device)
    neighbors[tuple(graph.edge_index)] = 1
    degree = neighbors.sum(dim=1)
    twice_triangles = ((neighbors @ neighbors) * neighbors).sum(dim=1)
    clustering = twice_triangles / (degree * (degree - 1)).clamp_min(1)
    return torch.stack((degree.log1p(), clustering, (twice_triangles / 2).log1p()), dim=-1)


def greedy(graph, score):
    u, v = graph.edge_index.cpu()
    neighbors = [set() for _ in range(graph.num_nodes)]
    for a, b in zip(u.tolist(), v.tolist()):
        neighbors[a].add(b)
    order = score.flatten().argsort(descending=True).tolist()
    best = set()
    for seed in range(min(5, len(order))):
        selected = set()
        for node in order[seed:]:
            if selected.isdisjoint(neighbors[node]):
                selected.add(node)
        if len(selected) > len(best):
            best = selected
    solution = torch.zeros(graph.num_nodes, dtype=torch.bool, device=score.device)
    solution[sorted(best)] = True
    return solution


class GCON(nn.Module):
    def __init__(self, hidden, layers):
        super().__init__()
        spec = importlib.util.spec_from_file_location("upstream_gcon", SOURCE / "graphgym/layer/hybridconv_v2.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.encode = nn.Linear(3, hidden)
        self.layers = nn.ModuleList(module.HybridConv_v2(hidden, hidden, norm="rw") for _ in range(layers))
        self.output = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = self.encode(x).relu()
        for layer in self.layers:
            x = x + layer(x, edge_index)
        return self.output(x).squeeze(-1).sigmoid()


def mis_loss(batch, p):
    groups = batch.num_graphs
    u, v = batch.edge_index
    edges = scatter(p[u] * p[v], batch.batch[u], dim_size=groups, reduce="sum")
    cardinality = scatter(p, batch.batch, dim_size=groups, reduce="sum")
    squares = scatter(p.square(), batch.batch, dim_size=groups, reduce="sum")
    return ((1.1 * edges - cardinality.square() + squares) * batch.ptr.diff()).mean()


def train(args, model, device):
    data = ROOT / "models/data/MIS_tensors"
    train = list(Dataset(data / "RB_small_train", torch.float32, args.train_samples))
    val = list(Dataset(data / "RB_small_val", torch.float32, args.valid_samples))
    for graph in train + val:
        graph.x = features(graph)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    best = -1
    checkpoint = args.result_dir / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            loss = mis_loss(batch, model(batch.x, batch.edge_index))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            size = 0
            for graph in val:
                graph = graph.clone().to(device)
                size += greedy(graph, model(graph.x, graph.edge_index)).sum().item()
            size /= len(val)
        if size > best:
            best = size
            torch.save(model.state_dict(), checkpoint / "best.pt")
        print(f"Epoch {epoch}: validation size {size:.3f}", flush=True)
    metadata = dict(version="v1", status="completed", mode="train", run=args.result_dir.name,
                    method="GCON", config=vars(args), started_at=started_at,
                    wall_time_s=perf_counter() - started, best_val_size=best)
    (args.result_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")


def main():
    parser = add_evaluation_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--valid_samples", type=int, default=100)
    parser.add_argument("--test_samples", type=int, default=-1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--test_path", default=str(TEST))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = GCON(args.hidden, args.layers).to(device)
    if not args.checkpoint:
        train(args, model, device)
        return
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location=device))
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    def solve(graph, generator, index):
        return greedy(graph, model(features(graph), graph.edge_index)), {}

    evaluate(args, solve, device, params)


if __name__ == "__main__":
    main()
