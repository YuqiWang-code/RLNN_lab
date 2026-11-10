"""Measure node-aligned context and rollout stability after relabeling."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from eval_v2 import TRAIN, VALID, dataset_at, write_run
from mis_v1 import adjacency, best_decoded
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import load_pave_v3
from v7_runs import pave, preflight_policy, provenance


def compare(a, b):
    residual = (a-b).float()
    denom = a.float().std().clamp_min(1e-8)
    top = max(1, a.shape[0] // 10)
    first = a.topk(top, dim=0).indices
    second = b.topk(top, dim=0).indices
    overlap = torch.stack([torch.isin(first[:, c], second[:, c]).float().mean() for c in range(a.shape[1])])
    return dict(bias_mae=float(residual.abs().mean()),
                bias_relative_rmse=float(residual.square().mean().sqrt()/denom),
                bias_cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
                top_decile_overlap=float(overlap.mean()))


def rollout(solver, graph, adj, topology, initial, noise, legal, ranker):
    x = initial
    best = x
    energy = solver.problem.energy_func(adj, graph.b, x)[0]
    for uniform in noise:
        probability = solver.flip_logits(x, graph, adj, topology).sigmoid()
        x = torch.where(uniform < probability, 1-x, x)
        value = solver.problem.energy_func(adj, graph.b, x)[0]
        best = torch.where(value[None] < energy[None], x, best)
        energy = torch.minimum(energy, value)
    decoded = best_decoded(solver.problem, best, adj, graph.b)
    selected = sum(1 << i for i in decoded.nonzero().flatten().tolist())
    assert legal.feasible(selected)
    final = platform_search(legal, selected, choose=ranker.choose, horizon=1).best
    assert legal.feasible(final)
    return selected.bit_count(), final.bit_count()


@torch.no_grad()
def diagnose(args):
    policy = preflight_policy(args.arm, "no", ref=args.ref, width=args.width, checkpoint=args.checkpoint)
    base = preflight_policy("none", "no", ref=args.ref) if args.checkpoint else None
    policy.order_views = args.views
    device = policy.accelerator.device
    model = policy.accelerator.unwrap_model(policy.model)
    ranker = load_pave_v3(pave(args.ref)) if args.checkpoint else None
    data = dataset_at(TRAIN if args.checkpoint is None else VALID, policy.dtype, args.samples)
    random = np.random.default_rng(7311)
    rows = []
    for index, graph in enumerate(data):
        gid = Path(data.files[index]).stem
        seed = 1017 + index
        legal = SearchGraph(graph.num_nodes, graph.edge_index, seed=seed)
        graph = graph.to(device)
        adj = adjacency(graph)
        generator = torch.Generator(device=device).manual_seed(seed)
        initial = torch.randint(0, 2, (graph.num_nodes, 20), device=device, generator=generator).float()
        noise = torch.rand((100, *initial.shape), device=device, generator=generator)
        original_seed = policy.order_seed
        for variant in range(args.relabels):
            relabel_seed = int(random.integers(1, 2**31))
            perm = torch.as_tensor(np.random.default_rng(relabel_seed).permutation(graph.num_nodes), device=device)
            inverse = perm.argsort()
            relabeled = graph.clone()
            relabeled.edge_index = perm[graph.edge_index]
            relabel_adj = adjacency(relabeled)
            relabel_legal = SearchGraph(graph.num_nodes, relabeled.edge_index.cpu(), seed=seed)
            policy.order_seed = original_seed
            original = policy.policy_topology(graph, adj)
            original_bias = model.adapter.precompute(original)
            original_logits = policy.flip_logits(initial, graph, adj, original)
            for regime in ("fixed", "changed"):
                policy.order_seed = original_seed if regime == "fixed" else original_seed + 100 + variant
                topo = policy.policy_topology(relabeled, relabel_adj)
                bias = model.adapter.precompute(topo)[perm]
                logits = policy.flip_logits(initial[inverse], relabeled, relabel_adj, topo)[perm]
                row = dict(graph_id=gid, relabel_seed=relabel_seed, order_seed=policy.order_seed, regime=regime,
                           logit_mae=float((logits-original_logits).abs().mean()),
                           probability_mae=float((logits.sigmoid()-original_logits.sigmoid()).abs().mean()),
                           flip_disagreement=float(((noise[0] < logits.sigmoid()) !=
                                                    (noise[0] < original_logits.sigmoid())).float().mean()),
                           **compare(original_bias, bias))
                if args.checkpoint:
                    for label, solver in (("context", policy), ("baseline", base)):
                        p = original if label == "context" else solver.policy_topology(graph, adj)
                        q = topo if label == "context" else solver.policy_topology(relabeled, relabel_adj)
                        a3, final = rollout(solver, graph, adj, p, initial, noise, legal, ranker)
                        renamed_a3, renamed_final = rollout(solver, relabeled, relabel_adj, q, initial[inverse],
                                                             noise[:, inverse], relabel_legal, ranker)
                        row.update({label+"_delta_a3": renamed_a3-a3, label+"_delta_pave": renamed_final-final})
                rows.append(row)
            policy.order_seed = original_seed
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per_relabel.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {regime: {key: mean(row[key] for row in rows if row["regime"] == regime)
                        for key in rows[0] if key not in ("graph_id", "relabel_seed", "order_seed", "regime")}
               for regime in ("fixed", "changed")}
    write_run(args.output, dict(**provenance(args.ref), status="completed", arm=args.arm, samples=len(data),
                               order_views=args.views, checkpoint=args.checkpoint, summary=summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="v7_mamba")
    parser.add_argument("--ref", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--relabels", type=int, default=3)
    parser.add_argument("--views", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    diagnose(parser.parse_args())
