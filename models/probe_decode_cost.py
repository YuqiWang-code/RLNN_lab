"""Attribute the cost of the A.3 decode, now that it dominates the captured arm.

``outputs/v8_macro/positioning/capture_budget/capture_budget_ref0.csv`` says the
decode is 71.5 % of search time at 0.5 s and 86.6 % at 1.0 s on the captured arm:
the block search got so cheap that A.3 became the bottleneck.  Before rewriting
it, measure where the per-decode milliseconds actually go.

``mis.Problem.decode_gradient_flip`` is a host-driven greedy loop:

    while True:
        gradient = 2*beta*(A @ x) + b
        delta = gradient * (2x - 1) / 2
        gain, node = delta.max(dim=0)
        if not (gain > 0).any():        # <- one device-to-host read per iteration
            break
        x[node, particles] = where(...)

so each iteration ends in a full pipeline flush.  This script counts the
iterations, times the GPU work of one iteration without that read, times the
read itself, and times what a fixed-size chunk of the same body costs once it is
captured as a CUDA graph.  Those four numbers decide the rewrite: if the read
dominates, a chunked captured decoder is a pure win; if the sparse matmul
dominates, no amount of capture helps and the loop has to change shape.

    python -u models/probe_decode_cost.py --reference_seed 0 \
        --save_dir outputs/v2/none_seed0/checkpoints --split validation \
        --graphs 3 --repeat 5 --result_dir outputs/v8_macro/decode_cost
"""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from diagnose_v7_census import build_parser, solution_masks
from eval_v2 import dataset_at, load_policy, sync
from mis_v1 import adjacency
from v7_runs import provenance

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 8


def build_probe_parser():
    parser = build_parser()
    parser.add_argument("--graphs", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--search_steps", type=int, nargs="+", default=[1, 25, 100, 300],
                        help="decode after this many frozen steps; 1 is the initial state")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.set_defaults(result_dir=ROOT / "outputs/v8_macro/decode_cost")
    return parser


def masks_of(states):
    """Integer bitsets of every particle column, packed the same way as ``solution_masks``."""
    packed = np.packbits(states.t().to(torch.uint8).cpu().numpy(), axis=1, bitorder="little")
    return [int.from_bytes(row.tobytes(), "little") for row in packed]


def timed(function, device, repeat=1):
    times = []
    for _ in range(repeat):
        sync(device)
        tick = perf_counter()
        function()
        sync(device)
        times.append(perf_counter() - tick)
    return median(times)


def count_iterations(problem, states, adj, bias, limit=100_000):
    """Iterations the frozen A.3 loop takes from ``states``, and its wall time."""
    solutions = states.clone()
    particles = torch.arange(solutions.size(1), device=solutions.device)
    sync(solutions.device)
    tick = perf_counter()
    iterations = 0
    while True:
        gradient = 2 * problem.beta * (adj @ solutions) + bias
        delta = gradient * (2 * solutions - 1) / 2
        gain, node = delta.max(dim=0)
        improves = gain > 0
        if not improves.any():
            break
        selected = solutions[node, particles]
        solutions[node, particles] = torch.where(improves, 1 - selected, selected)
        iterations += 1
        if iterations >= limit:
            break
    sync(solutions.device)
    return iterations, perf_counter() - tick, solutions


def iteration_body(problem, adj, bias, solutions, particles, workspace):
    """One A.3 iteration with fixed control flow: safe to capture."""
    gradient, delta, gain, node, updates = workspace
    torch.add(torch.mul(adj @ solutions, 2 * problem.beta), bias, out=gradient)
    torch.mul(gradient, (2 * solutions - 1) / 2, out=delta)
    torch.max(delta, dim=0, out=(gain, node))
    selected = solutions.gather(0, node.unsqueeze(0)).squeeze(0)
    torch.where(gain > 0, 1 - selected, selected, out=updates)
    solutions.index_put_((node, particles), updates)
    return gain


def capture_chunk(problem, adj, bias, solutions, particles, chunk):
    """Capture ``chunk`` A.3 iterations; the replay also reports whether any flipped."""
    workspace = (torch.empty_like(solutions), torch.empty_like(solutions),
                 torch.empty(solutions.size(1), device=solutions.device, dtype=solutions.dtype),
                 torch.empty(solutions.size(1), device=solutions.device, dtype=torch.long),
                 torch.empty(solutions.size(1), device=solutions.device, dtype=solutions.dtype))
    flag = torch.zeros(1, device=solutions.device, dtype=torch.bool)

    def body():
        for _ in range(chunk):
            gain = iteration_body(problem, adj, bias, solutions, particles, workspace)
            torch.logical_or(flag, (gain > 0).any(dim=0, keepdim=True), out=flag)

    device = solutions.device
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        body()
    torch.cuda.current_stream(device).wait_stream(stream)
    executor = torch.cuda.CUDAGraph()
    with torch.cuda.graph(executor, stream=stream):
        body()
    torch.cuda.current_stream(device).wait_stream(stream)
    sync(device)
    return executor, flag


def replay_to_convergence(executor, flag, limit):
    """Replay captured chunks until one reports no flip."""
    replays = 0
    while True:
        flag.zero_()
        executor.replay()
        replays += 1
        if not bool(flag.item()):
            break
        if replays >= limit:
            break
    return replays


def measure_graph(solver, dataset, index, args):
    problem = solver.problem
    device = solver.accelerator.device
    graph = dataset[index].to(device)
    adj = adjacency(graph)
    bias = graph.b
    normalized = solver.policy_topology(graph, adj)
    generator = torch.Generator(device=device).manual_seed(1017)
    solver.num_t = max(args.search_steps) + 8
    states = solver.sample_states(graph, adj, generator=generator, normalized=normalized)
    first = next(states)
    walked = [first]
    for _ in range(max(args.search_steps) - 1):
        walked.append(next(states))

    rows = []
    for steps in args.search_steps:
        states_at = walked[min(steps, len(walked)) - 1]
        iterations, decode_seconds, decoded = count_iterations(problem, states_at, adj, bias)
        masks = solution_masks(problem, states_at, adj, bias)
        solutions = states_at.clone()
        particles = torch.arange(solutions.size(1), device=device)
        workspace = (torch.empty_like(solutions), torch.empty_like(solutions),
                     torch.empty(solutions.size(1), device=device, dtype=solutions.dtype),
                     torch.empty(solutions.size(1), device=device, dtype=torch.long),
                     torch.empty(solutions.size(1), device=device, dtype=solutions.dtype))
        core = timed(lambda: iteration_body(problem, adj, bias, solutions, particles, workspace),
                     device, args.repeat)
        sparse = timed(lambda: adj @ solutions, device, args.repeat)
        sync_once = timed(lambda: sync(device), device, args.repeat)
        masks_time = timed(lambda: solution_masks(problem, states_at, adj, bias), device, args.repeat)

        executor, flag = capture_chunk(problem, adj, bias, solutions, particles, args.chunk)

        def captured_decode():
            solutions.copy_(states_at)
            return replay_to_convergence(executor, flag, limit=args.chunk * 4096)

        # The captured path must land on the same integer bitsets as ``masks``.
        replays = captured_decode()
        captured_masks = masks_of(solutions)
        captured = timed(captured_decode, device, args.repeat)
        solutions.copy_(states_at)
        chunk_only = timed(lambda: executor.replay(), device, args.repeat)
        rows.append(dict(graph=f"idx_{index}", nodes=int(graph.num_nodes), search_steps=steps,
                         iterations=iterations,
                         decode_ms=round(1e3 * decode_seconds, 3),
                         masks_ms=round(1e3 * masks_time, 3),
                         per_iteration_ms=round(1e3 * decode_seconds / max(iterations, 1), 3),
                         gpu_core_ms=round(1e3 * core, 4),
                         sparse_mm_ms=round(1e3 * sparse, 4),
                         sync_ms=round(1e3 * sync_once, 4),
                         captured_replays=replays,
                         captured_chunk_ms=round(1e3 * chunk_only, 4),
                         captured_total_ms=round(1e3 * captured, 4),
                         captured_matches=bool(captured_masks == masks)))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        del executor, flag
        sync(device)
    return rows


def main():
    parser = build_probe_parser()
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    solver = load_policy(args)
    solver.model.requires_grad_(False)
    dataset = dataset_at(args.test_path, solver.dtype, args.graphs, official=False)
    rows = []
    for index in range(len(dataset)):
        rows.extend(measure_graph(solver, dataset, index, args))
    payload = dict(version="v8_positioning", mode="decode_cost", rows=rows,
                   config=vars(args), **provenance(args.reference_seed))
    path = args.result_dir / f"decode_cost_ref{args.reference_seed}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    totals = [r for r in rows if r["search_steps"] == max(args.search_steps)]
    print(f"at search_steps={max(args.search_steps)}: "
          f"decode {median(r['decode_ms'] for r in totals):.3f} ms over "
          f"{median(r['iterations'] for r in totals):.0f} iterations "
          f"({median(r['per_iteration_ms'] for r in totals):.3f} ms/iteration) vs "
          f"captured {median(r['captured_total_ms'] for r in totals):.3f} ms "
          f"({median(r['captured_replays'] for r in totals):.0f} replays of "
          f"{args.chunk}, {median(r['captured_chunk_ms'] for r in totals):.3f} ms/chunk), "
          f"matches {all(r['captured_matches'] for r in rows)}", flush=True)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
