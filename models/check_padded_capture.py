"""Gate for the amortised padded engine (``models/mis_v8_pad.py``).

Three things have to hold before any timing is believed:

1. the padded buffers reproduce the *unpadded* frozen trajectory bit for bit --
   same seed, same number of steps, same archive;
2. the captured replay reproduces the padded eager loop bit for bit (this is what
   proves the per-graph buffer rewrite is picked up by the recorded graph);
3. the packed decode lands on the same bitset as ``solution_masks``.

It also reports the two costs that decide the paper's clock: the per-graph load
(paid outside the clock) and the replay step time.

    python -u models/check_padded_capture.py --split validation --test_samples 4 \
        --steps 60 --result_dir outputs/v8_macro/padded
"""
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from diagnose_v7_census import solution_masks
from eval_v2 import dataset_at, load_policy, sync
from eval_v8_capture_budget import build_capture_parser
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v7_recombine import is_independent
from mis_v8_pad import PaddedSearch
from v7_runs import provenance


def reference(solver, graph, adj, normalized, seed, steps):
    """The unpadded frozen loop, line for line the deployed one."""
    device = graph.b.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    state = torch.randint(0, 2, (graph.num_nodes, solver.num_k), device=device,
                          dtype=solver.dtype, generator=generator)
    best = state.clone()
    energy = solver.problem.energy_func(adj, graph.b, state)[0]
    best_energy = energy.clone()
    for _ in range(steps):
        logits = solver.flip_logits(state, graph, adj, normalized)
        noise = torch.rand(state.shape, device=device, dtype=solver.dtype, generator=generator)
        state = torch.where(noise < logits.sigmoid(), 1 - state, state)
        energy = solver.problem.energy_func(adj, graph.b, state)[0]
        improves = energy < best_energy
        best = torch.where(improves.unsqueeze(0), state, best)
        best_energy = torch.where(improves, energy, best_energy)
    return best, best_energy


def shape_of(dataset, limit=None):
    """Largest node count and largest directed degree over the graphs we will run."""
    nodes = degree = 0
    for index in range(len(dataset) if limit is None else min(limit, len(dataset))):
        edges = dataset[index].edge_index
        nodes = max(nodes, int(edges.max()) + 1, int(dataset[index].num_nodes))
        counts = torch.bincount(edges[0], minlength=int(dataset[index].num_nodes))
        degree = max(degree, int(counts.max()))
    return nodes, degree + 1


def main():
    parser = build_capture_parser()
    parser.add_argument("--graphs", type=int, default=None, help="cap the shape scan")
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    output = args.result_dir / "check"
    output.mkdir(parents=True, exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    indices = list(range(len(dataset)))[args.shard::args.shards]
    n_max, capacity = shape_of(dataset, args.graphs)
    print(f"padded shape: nodes {n_max}, capacity {capacity}", flush=True)

    engine = PaddedSearch(solver, n_max=n_max, capacity=capacity, chunk=args.chunk,
                          device=device, decode_chunk=args.decode_chunk)
    sync(device)

    rows = []
    for position, index in enumerate(indices):
        graph = dataset[index]
        edges = graph.edge_index
        graph = graph.to(device)
        adj = adjacency(graph)
        n = int(graph.num_nodes)
        for seed in args.seeds:
            topology = SearchGraph(n, edges, seed=seed)
            normalized = solver.policy_topology(graph, adj)
            best, best_energy = reference(solver, graph, adj, normalized, seed, args.steps)

            loaded = engine.load(graph, edges, seed)
            sync(device)
            _, eager_best, eager_energy = engine.eager_step(args.steps)
            sync(device)
            # The padded row carries ``capacity`` slots where the unpadded row
            # carries ``degree + 1``.  Zero weights add nothing, but ``spmm``
            # reduces the non-zero terms in a different order, so the two
            # operators differ in the last bits (measured ~1e-7 per layer).  The
            # padded engine is therefore the reference for exactness, and the
            # agreement with the unpadded loop is reported as a distance.
            reference_masks = solution_masks(solver.problem, best, adj, graph.b)
            reference_size = max(reference_masks, key=int.bit_count).bit_count()
            drifted = int((eager_best[:n] != best).sum())
            energy_gap = abs(float(eager_energy.reshape(-1).min()) - float(best_energy.reshape(-1).min()))

            engine.load(graph, edges, seed)
            sync(device)
            decoded = []

            def decode(states):
                decoded.append(engine.decode_best(states))

            started = perf_counter()
            steps, decode_seconds, blocks = engine.run(float("inf"), decode,
                                                       max_blocks=args.steps // args.chunk)
            captured = perf_counter() - started
            sync(device)
            assert torch.equal(engine.best_states[:n], eager_best[:n]), (
                f"{index}: captured replay diverged from the padded eager loop")

            eager_masks = solution_masks(solver.problem, eager_best[:n], adj, graph.b)
            incumbent = max(eager_masks, key=int.bit_count)
            assert decoded and decoded[-1] == incumbent, (
                f"{index}: decoded bitset differs from solution_masks")
            assert is_independent(incumbent, topology.neighbors), f"{index}: illegal set"
            assert abs(incumbent.bit_count() - reference_size) <= 1, (
                f"{index}: padding changed the size ({reference_size} -> "
                f"{incumbent.bit_count()})")

            # Replay cost on its own, without the decode in the loop.
            engine.load(graph, edges, seed)
            engine.run(float("inf"), lambda states: None, max_blocks=args.steps // args.chunk)
            sync(device)
            started = perf_counter()
            engine.run(float("inf"), lambda states: None, max_blocks=args.steps // args.chunk)
            sync(device)
            replay_seconds = perf_counter() - started
            rows.append(dict(graph=index, nodes=n, seed=seed, load_ms=round(1e3 * loaded, 2),
                             steps=steps, blocks=blocks,
                             ms_per_step=round(1e3 * replay_seconds / max(steps, 1), 4),
                             decode_ms=round(1e3 * decode_seconds / max(len(decoded), 1), 3),
                             size=incumbent.bit_count(), reference_size=reference_size,
                             drifted_nodes=drifted, energy_gap=round(energy_gap, 6)))
            print(f"{position + 1}/{len(indices)} {index}: capture identical over {steps} steps "
                  f"(vs unpadded: {drifted} node-slots drifted, size "
                  f"{reference_size} -> {incumbent.bit_count()}), load "
                  f"{1e3 * loaded:.1f} ms, replay "
                  f"{1e3 * replay_seconds / max(steps, 1):.3f} ms/step, decode "
                  f"{1e3 * decode_seconds / max(len(decoded), 1):.2f} ms", flush=True)

    summary = dict(n_max=n_max, capacity=capacity, steps=args.steps, chunk=args.chunk,
                   graphs=len(indices), seeds=list(args.seeds),
                   load_ms=mean(r["load_ms"] for r in rows),
                   ms_per_step=mean(r["ms_per_step"] for r in rows),
                   decode_ms=mean(r["decode_ms"] for r in rows),
                   capture_s=None, rows=rows)
    tag = "" if args.shards == 1 else f"_shard{args.shard}"
    (output / f"check_ref{args.reference_seed}{tag}.json").write_text(
        json.dumps(dict(summary=summary, config=vars(args),
                        **provenance(args.reference_seed)), indent=2, default=str) + "\n",
        encoding="utf-8")
    print("padded engine gate passed: bit-identical trajectory, replay and decode", flush=True)


if __name__ == "__main__":
    main()
