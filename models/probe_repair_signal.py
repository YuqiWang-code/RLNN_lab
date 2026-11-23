"""Positive-sample gate for a learned repair head (v10 §5).

For every graph we first run the frozen policy from a random state to an
incumbent, then repeat R rounds of the following cycle:

    pick a selected node -> BFS ball of radius r -> evict every node in the ball
    -> repair for K steps with the ball frozen out of the solution
    -> A.3 decode -> delta = size_after - size_before

delta > 0 is the only training signal a repair head could learn from, so the
whole line is gated on the frequency of delta > 0.  The gate is 1%: at 20k
rounds we need >200 wins.  If the evicted ball is never compensated -- the
policy cannot place points outside it that make the solution larger -- there is
nothing to learn and the move-set line closes.
"""
import argparse
import json
import random
from pathlib import Path

import torch

import mis_v1
from eval_v2 import dataset_at, load_policy, policy_parser
from mis_v3_search import SearchGraph

DEFAULT_POLICY = Path("/home/hzeng/project/ZH/RLNN_lab/outputs/v2/none_seed0/checkpoints")
DEFAULT_TEST = "models/data/MIS_tensors/RB_small_author_test"


def ball(neighbours, seed, radius):
    """Bitset BFS ball of `radius` hops around `seed`, seed included."""
    seen = 1 << seed
    frontier = seen
    for _ in range(radius):
        nxt = 0
        rest = frontier
        while rest:
            bit = rest & -rest
            rest ^= bit
            nxt |= neighbours[bit.bit_length() - 1]
        nxt &= ~seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def indices(mask, num_nodes):
    return [node for node in range(num_nodes) if mask >> node & 1]


def run_steps(solver, graph, adjacency, topology, x, steps, blocked, generator,
              release_after=None):
    """Advance the frozen policy `steps` times and return its best-energy column.

    While `step < release_after` the `blocked` rows are forced to 0 with flip
    probability -inf, so the repair has to place points somewhere else.  After
    that the ball is released again (tabu expiry), which is what makes a larger
    solution reachable at all: with a permanent block the evicted ball can never
    be reconstructed, so delta > 0 is impossible by construction.
    """
    rows = None
    if blocked:
        rows = torch.zeros(x.shape[0], 1, device=x.device, dtype=torch.bool)
        rows[blocked] = True
        x = x.clone()
        x[blocked, :] = 0.0
    best_energy = None
    best_state = x[:, 0].clone()
    for step in range(steps):
        if rows is not None and release_after is not None and step >= release_after:
            rows = None
        logits = solver.flip_logits(x, graph, topology, topology)
        if rows is not None:
            logits = logits.masked_fill(rows, torch.finfo(logits.dtype).min)
        draw = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        x = torch.where(draw < torch.sigmoid(logits), 1 - x, x)
        if rows is not None:
            x = x.masked_fill(rows, 0.0)
        energy, _ = solver.problem.energy_func(adjacency, graph.b, x, False)
        column = int(energy.argmin())
        value = float(energy[column])
        if best_energy is None or value < best_energy:
            best_energy = value
            best_state = x[:, column].clone()
    return best_state


def decode_size(solver, adjacency, bias, state):
    solution = solver.problem.decode_gradient_flip(state.unsqueeze(1), adjacency, bias,
                                                   return_solutions=True)
    return int(solution[:, 0].sum().item()), solution[:, 0]


def main():
    parser = policy_parser()
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--warm_steps", type=int, default=400)
    parser.add_argument("--warm_particles", type=int, default=4,
                        help="particles for the warm start; the repair uses --num_k")
    parser.add_argument("--repair_steps", type=int, default=200)
    parser.add_argument("--release_steps", type=int, default=50)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.set_defaults(save_dir=str(DEFAULT_POLICY), test_path=DEFAULT_TEST,
                        mixed_precision="no", num_workers=0,
                        result_dir=Path("outputs/v8_macro/repair_signal"))
    args = parser.parse_args()
    if args.test_samples <= 0:
        args.test_samples = 100

    solver = load_policy(args)
    device = solver.accelerator.device
    dtype = solver.dtype
    particles = solver.num_k
    dataset = dataset_at(args.test_path, dtype, args.test_samples * args.shards, official=True)
    mine = list(range(len(dataset)))[args.shard::args.shards]

    rng = random.Random(args.seed + args.shard)
    rows = []
    for index in mine:
        graph = dataset[index]
        n = int(graph.num_nodes)
        edges = graph.edge_index
        graph = graph.to(device)
        adjacency = mis_v1.adjacency(graph)
        topology = solver.policy_topology(graph, adjacency)
        neighbours = SearchGraph(n, edges, seed=args.seed).neighbors

        generator = torch.Generator(device=device).manual_seed(args.seed + 17 * index)
        start = torch.randint(0, 2, (n, args.warm_particles), device=device, dtype=dtype,
                              generator=generator)
        warm = run_steps(solver, graph, adjacency, topology, start, args.warm_steps, [], generator)
        size_now, incumbent = decode_size(solver, adjacency, graph.b, warm)

        for step in range(args.rounds):
            radius = args.radii[step % len(args.radii)]
            selected = [node for node in range(n) if incumbent[node] > 0.5]
            if not selected:
                break
            centre = rng.choice(selected)
            region = ball(neighbours, centre, radius)
            evicted = indices(region, n)
            holed = incumbent.unsqueeze(1).repeat(1, particles).clone()
            repaired = run_steps(solver, graph, adjacency, topology, holed,
                                 args.repair_steps, evicted, generator,
                                 release_after=args.release_steps)
            size_after, vector = decode_size(solver, adjacency, graph.b, repaired)
            rows.append({
                "graph_id": f"idx_{index}",
                "round": step,
                "radius": radius,
                "evicted_nodes": len(evicted),
                "evicted_selected": sum(1 for node in evicted if incumbent[node] > 0.5),
                "size_before": size_now,
                "size_after": size_after,
                "delta": size_after - size_now,
            })
            if size_after > size_now:
                size_now = size_after
                incumbent = vector

    args.result_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.shards == 1 else f"_shard{args.shard}"
    with (args.result_dir / f"repair_signal{suffix}.json").open("w") as handle:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "rows": rows}, handle)
    total = len(rows)
    positive = sum(1 for row in rows if row["delta"] > 0)
    mean_delta = sum(row["delta"] for row in rows) / max(total, 1)
    print(f"rounds {total} positive {positive} rate {positive / max(total, 1):.4f} "
          f"mean_delta {mean_delta:+.4f} max_delta {max((r['delta'] for r in rows), default=0)}")


if __name__ == "__main__":
    main()
