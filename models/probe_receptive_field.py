"""G1: how far does the frozen policy actually look?

The v9 direction rests on one measured claim: the five-layer backbone behaves like a
one-hop rule, so two states with the same ``(x_i, selected neighbours of i)`` get the
same logit and the 20 search particles share a single fixed point.  This probe
measures it directly on the frozen checkpoint, without training anything:

1. ``R2`` of the logits on ``[1, x, c1]``, then adding ``c2``, then ``c3``, where
   ``c_k`` is the number of selected nodes within ``k`` hops of ``i``.  A five-layer
   mean aggregator should saturate at ``c1``.
2. the influence profile: flip *every* node at exact distance ``k`` from a target and
   record ``|dlogit|`` at the target, for ``k = 1..5``.  The architecture can reach
   five hops, so the question is what each hop is worth, not whether it is wired.
3. a control: flipping nodes farther than ``hops`` from a target must change nothing.

    CUDA_VISIBLE_DEVICES=0 python -u models/probe_receptive_field.py \
        --save_dir outputs/v2/none_seed0/checkpoints \
        --test_path models/data/MIS_tensors/RB_small_author_test \
        --graphs 10 --targets 20 --hops 5 --result_dir outputs/v8_macro/receptive_field
"""
import json
from pathlib import Path

import torch

from eval_v2 import dataset_at, load_policy, policy_parser
from mis_v1 import adjacency


def hop_maps(dense, hops):
    """Return ``(exact, ball)``: exact-distance sets ``1..hops`` plus ``beyond``, and balls."""
    nodes = dense.shape[0]
    seen = torch.eye(nodes, dtype=torch.bool, device=dense.device)
    frontier = torch.eye(nodes, dtype=torch.bool, device=dense.device)
    exact, ball = [None], []
    for _ in range(hops):
        frontier = (frontier.float() @ dense.float()) > 0
        step = frontier & ~seen
        seen = seen | step
        exact.append(step)
        ball.append(seen.clone())
    exact.append(~seen)
    return exact, ball


def ols_r2(features, target):
    """Coefficient of determination of a least-squares fit with an intercept.

    Columns are standardised first: the ring counts are nested and highly
    collinear, and an unstandardised fit on raw counts overflows in float32.
    """
    features = features.double()
    spread = features.std(dim=0, keepdim=True).clamp_min(1e-8)
    features = (features - features.mean(dim=0, keepdim=True)) / spread
    design = torch.cat([torch.ones(features.shape[0], 1, device=features.device,
                                  dtype=features.dtype), features], dim=1)
    target = target.double()
    solution = torch.linalg.lstsq(design, target).solution
    residual = target - design @ solution
    total = ((target - target.mean()) ** 2).sum()
    return float(1.0 - (residual ** 2).sum() / total.clamp_min(1e-12))


def main():
    parser = policy_parser()
    parser.add_argument("--graphs", type=int, default=10)
    parser.add_argument("--targets", type=int, default=20)
    parser.add_argument("--hops", type=int, default=5)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--density", type=float, default=0.075)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    device = solver.accelerator.device
    generator = torch.Generator(device=device).manual_seed(args.seed)
    dataset = dataset_at(args.test_path, solver.dtype, args.graphs)

    rows = []
    influence = {hop: [] for hop in range(1, args.hops + 1)}
    beyond = []
    for index, graph in enumerate(dataset):
        graph = graph.to(device)
        adj = adjacency(graph)
        normalized = solver.policy_topology(graph, adj)
        nodes = graph.num_nodes
        edges = graph.edge_index
        dense = torch.zeros(nodes, nodes, dtype=torch.bool, device=device)
        dense[edges[0], edges[1]] = True
        exact, ball = hop_maps(dense, args.hops)

        with torch.no_grad():
            state = (torch.rand(nodes, args.particles, device=device,
                                generator=generator) < args.density).float()
            logits = solver.flip_logits(state, graph, adj, normalized)
            ring_hops = min(args.hops, 3)
            rings = [exact[hop].float() @ state for hop in range(1, ring_hops + 1)]
            r2 = []
            for hop in range(1, ring_hops + 1):
                # The node's own state plus the exact-k rings for k <= hop.
                features = torch.cat([state, *rings[:hop]], dim=1)
                r2.append(round(ols_r2(features.reshape(nodes, -1), logits.reshape(nodes, -1)), 4))
            rows.append({"graph": Path(dataset.files[index]).stem, "nodes": nodes,
                         "r2": r2, "logit_std": round(float(logits.std()), 4)})

            targets = torch.rand(nodes, device=device, generator=generator).argsort()[:args.targets].tolist()
            for hop in range(1, args.hops + 1):
                moved_logits = []
                for target in targets:
                    mask = exact[hop][target]
                    if not bool(mask.any()):
                        continue
                    flipped = state.clone()
                    flipped[mask] = 1.0 - flipped[mask]
                    moved = solver.flip_logits(flipped, graph, adj, normalized)
                    influence[hop].append(float((moved[target] - logits[target]).abs().mean()))
            for target in targets:
                mask = exact[-1][target]
                if not bool(mask.any()):
                    continue
                flipped = state.clone()
                flipped[mask] = 1.0 - flipped[mask]
                moved = solver.flip_logits(flipped, graph, adj, normalized)
                beyond.append(float((moved[target] - logits[target]).abs().max()))
        print(f"{index + 1}/{len(dataset)} {rows[-1]['graph']}: "
              f"R2 {rows[-1]['r2']} | hop influence "
              + " ".join(f"{hop}:{influence[hop][-1]:.5f}" for hop in influence
                         if influence[hop]), flush=True)

    summary = {
        "graphs": len(rows),
        "r2_mean": [round(sum(row["r2"][k] for row in rows if len(row["r2"]) > k) /
                          max(1, len(rows)), 4) for k in range(3)],
        "influence_mean": {str(hop): round(sum(values) / len(values), 6)
                           for hop, values in influence.items() if values},
        "beyond_max": max(beyond) if beyond else None,
        "rows": rows,
    }
    with (args.result_dir / "receptive_field.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps({key: summary[key] for key in ("graphs", "r2_mean", "influence_mean", "beyond_max")},
                     indent=2))


if __name__ == "__main__":
    main()
