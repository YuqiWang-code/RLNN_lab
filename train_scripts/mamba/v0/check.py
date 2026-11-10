"""Verify v7 warm start, graph isolation, gradients, and one-shot capture."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

import numpy as np
import torch

from eval_v2 import ROOT, TRAIN, dataset_at, search, write_run
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from torch_geometric.data import Data
from utils import build_batch
from v7_runs import preflight_policy, provenance
from v7_structure import structure


def graph_of(n, pairs, device, dtype):
    edges = [pair for a, b in pairs for pair in ((a, b), (b, a))]
    edge = torch.tensor(edges, device=device, dtype=torch.long).reshape(-1, 2).t().contiguous()
    return Data(num_nodes=n, edge_index=edge, edge_weight=torch.full((edge.size(1),), .5, dtype=dtype, device=device),
                x=torch.ones(n, 1, dtype=dtype, device=device), b=-torch.ones(n, 1, dtype=dtype, device=device),
                idx=0)


def check(args):
    mode = "v7_mamba" if args.device == "cuda" else "v7_mlp"
    base = preflight_policy("none", args.precision, cuda_graph=args.device == "cuda")
    solver = preflight_policy(mode, args.precision, cuda_graph=args.device == "cuda")
    device = solver.accelerator.device
    assert device.type == args.device
    model = solver.accelerator.unwrap_model(solver.model)
    original = base.accelerator.unwrap_model(base.model)
    for name, weight in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], weight, rtol=0, atol=0)
    context = model.adapter
    g = graph_of(6, [(0, 1), (0, 2), (1, 2), (2, 3)], device, solver.dtype)
    other = graph_of(3, [(0, 1), (1, 2)], device, solver.dtype)
    nodes = torch.tensor([[0, 1], [1, 0], [0, 1], [0, 0], [1, 0], [0, 1]],
                         device=device, dtype=solver.dtype)
    topo = solver.policy_topology(g, adjacency(g))
    with torch.no_grad():
        logits = solver.flip_logits(nodes, g, adjacency(g), topo)
        base_logits = base.flip_logits(nodes, g, adjacency(g), base.policy_topology(g, adjacency(g)))
        torch.testing.assert_close(logits, base_logits, atol=0, rtol=0)
        batch = build_batch([g, other])
        both = solver.policy_topology(batch, adjacency(batch))
        separate = torch.cat((context.precompute(topo),
                              context.precompute(solver.policy_topology(other, adjacency(other)))))
        torch.testing.assert_close(context.precompute(both), separate, atol=2e-6, rtol=1e-5)
        for n in (0, 1, 3):
            isolated = graph_of(n, [], device, solver.dtype)
            t = solver.policy_topology(isolated, adjacency(isolated))
            assert t.cached_action_bias.shape == (n, 2) and t.cached_action_bias.isfinite().all()
    # Tie randomness transported with the nodes is exactly relabel-consistent.
    edges = g.edge_index.cpu().numpy()
    perm = np.array([3, 5, 0, 2, 1, 4])
    keys = np.random.default_rng(141).random((2, 6))
    x, orders = structure(edges, 6, 0, order="bfs", tie_keys=keys)
    y, relabeled = structure(perm[edges], 6, 0, order="bfs", tie_keys=keys[:, np.argsort(perm)])
    np.testing.assert_array_equal(x, y[perm])
    assert {tuple(perm[row]) for group in orders for row in group} == {
        tuple(row) for group in relabeled for row in group}
    gradients = None
    if args.precision == "no":
        context.requires_grad_(True)
        model.train()
        optimizer = torch.optim.Adam(context.parameters(), lr=.001)
        train_topology = topo._replace(cached_action_bias=None)
        gradients = []
        for _ in range(2):
            optimizer.zero_grad()
            output = model(nodes, g.edge_index, adjacency(g), train_topology)
            output.sum().backward()
            gradients.append([float(context.alpha.grad.abs()), float(context.out_proj.weight.grad.norm())])
            optimizer.step()
        assert gradients[0][0] > 0 and gradients[1][1] > 0 and np.isfinite(gradients).all()
        model.eval()
        context.alpha.data.zero_()
    scan_shapes = []
    if args.device == "cuda":
        hook = context.sequence.register_forward_pre_hook(lambda _, args: scan_shapes.append(tuple(args[0].shape)))
    data = dataset_at(TRAIN, solver.dtype, 1)[0]
    legal = SearchGraph(data.num_nodes, data.edge_index, seed=1017)
    _, component_orders = structure(data.edge_index.numpy(), data.num_nodes, solver.order_seed + data.idx,
                                    order=context.order, views=solver.order_views)
    data = data.to(device)
    with torch.no_grad():
        b, _ = search(base, data, legal, 1017)
        s, steps = search(solver, data, legal, 1017)
    if args.device == "cuda":
        hook.remove()
        expected = [(2 * order.shape[0], order.shape[1], 32) for order in component_orders]
        assert scan_shapes == expected, (scan_shapes, expected)
    assert steps == 100 and legal.feasible(s)
    if args.precision == "no":
        assert s == b
    hashes = [provenance(ref)["pave_sha256"] for ref in range(3)]
    for ref, expected in enumerate(hashes):
        matches = (ROOT / "outputs/v5_frequency").glob(f"eval_budget_*_baseline_extended_seed{ref}/pave/run.json")
        archived = {json.loads(path.read_text(encoding="utf-8")).get("pave_sha256") for path in matches}
        assert expected in archived, f"Frozen PAVE checkpoint differs for ref {ref}"
    args.output.mkdir(parents=True, exist_ok=False)
    write_run(args.output, dict(**provenance(0), status="completed", arm=mode, precision=args.precision,
                                device=str(device), exact_zero_logits=True, capture_scan_shapes=scan_shapes,
                                gradient=gradients, baseline_size=b.bit_count(), v7_size=s.bit_count(),
                                same_solution=s == b, verified_pave_sha256=hashes))
    print(args.output, gradients, scan_shapes, b.bit_count(), s.bit_count(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--precision", choices=("no", "fp16"), default="no")
    parser.add_argument("--output", type=Path, required=True)
    check(parser.parse_args())
