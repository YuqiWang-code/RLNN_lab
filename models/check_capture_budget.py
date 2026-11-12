"""Gate for the captured budget search: identical trajectory, then speed.

A captured block is only allowed to be faster, never different.  The checks here
run the eager frozen loop and the captured loop from the same seed and the same
number of steps, and assert that the state, the per-particle archive and the A.3
decoded masks agree exactly.  Only then does the timing line mean anything.

    python -u models/check_capture_budget.py --reference_seed 0 \
        --save_dir outputs/v2/none_seed0/checkpoints --split validation \
        --graphs 2 --steps 20 --result_dir outputs/v8_macro/check_capture
"""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from diagnose_v7_census import build_parser, solution_masks
from eval_v2 import dataset_at, load_policy, sync
from mis_v1 import adjacency
from mis_v8_capture import CapturedSearch

ROOT = Path(__file__).resolve().parents[1]
DECODE_INTERVAL = 10


def build_check_parser():
    parser = build_parser()
    parser.add_argument("--graphs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--chunk", type=int, default=DECODE_INTERVAL)
    parser.add_argument("--repeat", type=int, default=3)
    parser.set_defaults(result_dir=ROOT / "outputs/v8_macro/check_capture")
    return parser


@torch.no_grad()
def eager_steps(solver, graph, adj, topology, seed, steps):
    device = graph.b.device
    generator = torch.Generator(device=device).manual_seed(seed)
    solver.num_t = steps * 4
    stream = solver.sample_states(graph, adj, generator=generator, normalized=topology)
    state = next(stream)
    best_states = state.clone()
    best_energy = solver.problem.energy_func(adj, graph.b, state)[0]
    for _ in range(steps):
        state = next(stream)
        energy = solver.problem.energy_func(adj, graph.b, state)[0]
        improves = energy < best_energy
        best_states = torch.where(improves.unsqueeze(0), state, best_states)
        best_energy = torch.where(improves, energy, best_energy)
    solver.num_t = 100
    return state, best_states, best_energy


@torch.no_grad()
def eager_timing(solver, graph, adj, topology, steps, repeat):
    device = graph.b.device
    times = []
    solver.num_t = steps * 2
    for _ in range(repeat):
        generator = torch.Generator(device=device).manual_seed(1017)
        stream = solver.sample_states(graph, adj, generator=generator, normalized=topology)
        next(stream)
        sync(device)
        tick = perf_counter()
        for _ in range(steps):
            state = next(stream)
            solver.problem.energy_func(adj, graph.b, state)
        sync(device)
        times.append((perf_counter() - tick) / steps)
    solver.num_t = 100
    return median(times)


def main():
    parser = build_check_parser()
    args = parser.parse_args()
    if args.steps % args.chunk:
        parser.error(f"--steps must be a multiple of --chunk ({args.chunk})")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "check"
    output.mkdir(exist_ok=True)

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, max(args.graphs, 1),
                         official=args.split == "test")
    rows = []
    for index, graph in enumerate(dataset):
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        graph = graph.to(device)
        adj = adjacency(graph)
        topology = solver.policy_topology(graph, adj)
        seed = 1017
        state, best_states, best_energy = eager_steps(solver, graph, adj, topology, seed, args.steps)
        eager_masks = solution_masks(solver.problem, best_states, adj, graph.b)

        sync(device)
        tick = perf_counter()
        search = CapturedSearch(solver, graph, adj, topology,
                                torch.Generator(device=device).manual_seed(seed), chunk=args.chunk)
        sync(device)
        capture_seconds = perf_counter() - tick

        decoded = []
        blocks = args.steps // args.chunk

        def decode(best):
            decoded.append(solution_masks(solver.problem, best, adj, graph.b))

        steps, decode_seconds, _ = search.run(perf_counter() + 1e6, decode, max_blocks=blocks)
        if steps != args.steps:
            raise AssertionError(f"{graph_id}: captured {steps} steps, expected {args.steps}")
        if not torch.equal(search.state, state):
            raise AssertionError(f"{graph_id}: state diverged (max |diff| "
                                 f"{float((search.state - state).abs().max())})")
        if not torch.equal(search.best_states, best_states):
            raise AssertionError(f"{graph_id}: archive states diverged")
        if not torch.equal(search.best_energy, best_energy):
            raise AssertionError(f"{graph_id}: archive energies diverged "
                                 f"(max |diff| {float((search.best_energy - best_energy).abs().max())})")
        captured_masks = decoded[-1]
        if [int(m) for m in captured_masks] != [int(m) for m in eager_masks]:
            raise AssertionError(f"{graph_id}: decoded masks diverged")

        # Skipping repeat decodes must leave the incumbent bit-for-bit unchanged.
        def incumbent_of(skip):
            engine = CapturedSearch(solver, graph, adj, topology,
                                    torch.Generator(device=device).manual_seed(seed),
                                    chunk=args.chunk)
            calls = []

            def keep(best):
                calls.append(max(solution_masks(solver.problem, best, adj, graph.b),
                                 key=int.bit_count))

            engine.run(perf_counter() + 1e6, keep, max_blocks=blocks, skip_unchanged=skip)
            return max(calls, key=int.bit_count), len(calls)

        full_incumbent, full_calls = incumbent_of(False)
        skipped_incumbent, skipped_calls = incumbent_of(True)
        if full_incumbent != skipped_incumbent:
            raise AssertionError(f"{graph_id}: skipping repeat decodes changed the incumbent")

        eager_seconds = eager_timing(solver, graph, adj, topology, args.steps, args.repeat)
        replay_times = []
        for _ in range(args.repeat):
            replay = CapturedSearch(solver, graph, adj, topology,
                                    torch.Generator(device=device).manual_seed(seed), chunk=args.chunk)
            sync(device)
            tick = perf_counter()
            replay.run(perf_counter() + 1e6, lambda best: None, max_blocks=blocks)
            sync(device)
            replay_times.append((perf_counter() - tick) / args.steps)
        row = dict(graph_id=graph_id, nodes=int(graph.num_nodes), steps=args.steps,
                   chunk=args.chunk, capture_ms=capture_seconds * 1e3,
                   eager_ms_per_step=eager_seconds * 1e3,
                   captured_ms_per_step=median(replay_times) * 1e3,
                   speedup=eager_seconds / median(replay_times),
                   decode_calls=full_calls, decode_calls_skipped=skipped_calls)
        rows.append(row)
        print(f"{graph_id}: identical over {args.steps} steps; eager {row['eager_ms_per_step']:.3f} ms, "
              f"captured {row['captured_ms_per_step']:.3f} ms ({row['speedup']:.2f}x), "
              f"capture {row['capture_ms']:.1f} ms, decodes {full_calls} -> {skipped_calls}", flush=True)

    summary = dict(graphs=len(rows), steps=args.steps, chunk=args.chunk,
                   eager_ms_per_step=median(row["eager_ms_per_step"] for row in rows),
                   captured_ms_per_step=median(row["captured_ms_per_step"] for row in rows),
                   capture_ms=median(row["capture_ms"] for row in rows))
    summary["speedup"] = summary["eager_ms_per_step"] / summary["captured_ms_per_step"]
    metadata = dict(version="v8_positioning", status="completed", mode="capture_gate",
                    checks=["state", "archive_states", "archive_energy", "decoded_masks"],
                    summary=summary, rows=rows, config=vars(args))
    (output / ("check_ref%d.json" % args.reference_seed)).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {output / ('check_ref%d.json' % args.reference_seed)}", flush=True)


if __name__ == "__main__":
    main()
