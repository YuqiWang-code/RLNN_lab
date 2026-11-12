"""Attribute the per-step cost of the deployment (budget) path.

The frozen deployment protocol searches until a wall-clock deadline, and every
millisecond it does not spend on search is a step it never takes.  The v7 budget
runs recorded ``ms_per_step`` per run: median 3.0-3.5 ms, i.e. only ~120 steps in
0.5 s (``outputs/v7_tccr/budget/budget_ref{0,1}.csv``).  The same sampling step
captured as a CUDA graph (``capture_mis.captured_states``, the fixed-work path
behind the v5 headline numbers) was measured at ~0.5 ms/step.

This script measures both paths back to back on one graph, one checkpoint, one
machine, and splits the eager cost into forward / draw / energy / decode, so the
gap is attributed rather than guessed.  If most of the gap is Python and launch
overhead, then chunking the search into captured blocks is a pure win: no change
to the policy, only more steps inside the same deadline.

    python -u models/profile_budget_step.py --reference_seed 0 \
        --save_dir outputs/v2/none_seed0/checkpoints --split validation \
        --graphs 3 --steps 200 --result_dir outputs/v8_macro/profile_step
"""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from capture_mis import captured_states
from diagnose_v7_census import build_parser, solution_masks
from eval_v2 import dataset_at, load_policy, sync
from mis_v1 import adjacency
from v7_runs import provenance

ROOT = Path(__file__).resolve().parents[1]


def build_profile_parser():
    parser = build_parser()
    parser.add_argument("--graphs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=3)
    parser.set_defaults(result_dir=ROOT / "outputs/v8_macro/profile_step")
    return parser


def timed(function, device, repeat=1):
    """Median wall time of ``repeat`` calls, synchronised on both ends."""
    times = []
    for _ in range(repeat):
        sync(device)
        tick = perf_counter()
        function()
        sync(device)
        times.append(perf_counter() - tick)
    return median(times)


def measure(solver, graph, adj, topology, steps, repeat):
    device = graph.b.device
    scale = 1e3 / steps
    generator = torch.Generator(device=device).manual_seed(1017)
    solver.num_t = steps * repeat + 8
    states = solver.sample_states(graph, adj, generator=generator, normalized=topology)
    first = next(states)

    def eager():
        local = first
        for _ in range(steps):
            local = next(states)
            solver.problem.energy_func(adj, graph.b, local)

    def captured_of(num_t):
        def run():
            solver.num_t = num_t
            for _ in captured_states(solver, graph, adj, topology, generator):
                pass
        return run

    def forward():
        for _ in range(steps):
            solver.flip_logits(first, graph, adj, topology)

    def energy():
        for _ in range(steps):
            solver.problem.energy_func(adj, graph.b, first)

    result = dict(
        eager_ms_per_step=timed(eager, device, repeat) * scale,
        captured_ms_per_step=timed(captured_of(steps - 1), device, repeat) * scale,
        forward_ms_per_step=timed(forward, device, repeat) * scale,
        energy_ms_per_step=timed(energy, device, repeat) * scale,
    )
    solver.num_t = 100
    result["decode_ms"] = timed(
        lambda: solution_masks(solver.problem, first, adj, graph.b), device, repeat) * 1e3
    result["eager_over_captured"] = result["eager_ms_per_step"] / result["captured_ms_per_step"]
    return result


def main():
    parser = build_profile_parser()
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "profile"
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
        # One throwaway pass so the allocator and kernel cache are warm, as in eval_v7_budget.
        solver.num_t = 20
        for _ in captured_states(solver, graph, adj, topology,
                                 torch.Generator(device=device).manual_seed(5)):
            pass
        solver.num_t = args.steps
        row = measure(solver, graph, adj, topology, args.steps, args.repeat)
        row.update(graph_id=graph_id, nodes=int(graph.num_nodes), steps=args.steps)
        rows.append(row)
        print(f"{graph_id}: eager {row['eager_ms_per_step']:.3f} ms/step, "
              f"captured {row['captured_ms_per_step']:.3f} ms/step "
              f"({row['eager_over_captured']:.2f}x), forward {row['forward_ms_per_step']:.3f}, "
              f"energy {row['energy_ms_per_step']:.3f}, decode {row['decode_ms']:.3f} ms", flush=True)

    summary = {key: median(row[key] for row in rows) for key in
               ("eager_ms_per_step", "captured_ms_per_step", "forward_ms_per_step",
                "energy_ms_per_step", "decode_ms", "eager_over_captured")}
    summary["steps_per_second_now"] = 1e3 / summary["eager_ms_per_step"]
    summary["steps_per_second_if_captured"] = 1e3 / summary["captured_ms_per_step"]
    summary["projection"] = {
        f"{budget:g}s": dict(
            now=round(budget * summary["steps_per_second_now"]),
            captured=round(budget * summary["steps_per_second_if_captured"]),
        ) for budget in (0.5, 1.0)
    }
    metadata = dict(version="v8_positioning", status="completed", mode="step_profile",
                    graphs=len(rows), steps=args.steps, repeat=args.repeat,
                    summary=summary, rows=rows, config=vars(args),
                    **provenance(args.reference_seed))
    (output / f"profile_ref{args.reference_seed}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {output / f'profile_ref{args.reference_seed}.json'}", flush=True)


if __name__ == "__main__":
    main()
