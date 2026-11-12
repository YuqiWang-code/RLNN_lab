"""Gate the captured A.3 decoder before any timing is believed.

Two things have to hold for ``mis_v8_decode.CapturedDecoder`` to replace
``Problem.decode_gradient_flip`` in the deployment path:

1. the integer bitsets must match the eager loop exactly, from the random initial
   state (where A.3 takes 110-130 iterations) as well as from the archive states
   of a run (6-10 iterations), and
2. it must be faster, otherwise there is nothing to buy.

The chunk length trades replay overhead against wasted no-op iterations: at 6-10
iterations typical, a chunk of 8 always burns a second replay.  This script
measures chunks 2/4/8/16 and prints the winner, which is the default the eval
script uses.

    python -u models/check_decode_capture.py --reference_seed 0 \
        --save_dir outputs/v2/none_seed0/checkpoints --split validation \
        --graphs 4 --repeat 5 --result_dir outputs/v8_macro/decode_cost
"""
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from diagnose_v7_census import build_parser, solution_masks
from eval_v2 import dataset_at, load_policy, sync
from mis_v1 import adjacency
from mis_v8_decode import CapturedDecoder
from probe_decode_cost import masks_of, timed
from v7_runs import provenance

CHUNKS = (2, 4, 8, 16)


def build_check_parser():
    parser = build_parser()
    parser.add_argument("--graphs", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--search_steps", type=int, nargs="+", default=[1, 25, 100, 300])
    return parser


def check_graph(solver, dataset, index, args):
    problem = solver.problem
    device = solver.accelerator.device
    graph = dataset[index].to(device)
    adj = adjacency(graph)
    normalized = solver.policy_topology(graph, adj)
    generator = torch.Generator(device=device).manual_seed(1017)
    solver.num_t = max(args.search_steps) + 8
    states = solver.sample_states(graph, adj, generator=generator, normalized=normalized)
    walked = [next(states)]
    for _ in range(max(args.search_steps) - 1):
        walked.append(next(states))

    rows = []
    for steps in args.search_steps:
        states_at = walked[min(steps, len(walked)) - 1]
        started = perf_counter()
        eager_masks = solution_masks(problem, states_at, adj, graph.b)
        eager_ms = 1e3 * timed(lambda: solution_masks(problem, states_at, adj, graph.b),
                               device, args.repeat)
        row = dict(graph=f"idx_{index}", nodes=int(graph.num_nodes), search_steps=steps,
                   eager_ms=round(eager_ms, 3))
        for chunk in CHUNKS:
            started = perf_counter()
            decoder = CapturedDecoder(problem, adj, graph.b, states_at, chunk=chunk)
            sync(device)
            capture_ms = 1e3 * (perf_counter() - started)
            captured = decoder.decode(states_at)
            captured_masks = masks_of(captured)
            assert captured_masks == eager_masks, (index, steps, chunk)
            decoder.decode(states_at)
            captured_ms = 1e3 * timed(lambda: decoder.decode(states_at), device, args.repeat)
            row[f"chunk{chunk}"] = dict(capture_ms=round(capture_ms, 3),
                                        decode_ms=round(captured_ms, 3),
                                        speedup=round(eager_ms / captured_ms, 2))
            del decoder
            sync(device)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def main():
    parser = build_check_parser()
    args = parser.parse_args()
    solver = load_policy(args)
    solver.model.requires_grad_(False)
    dataset = dataset_at(args.test_path, solver.dtype, args.graphs, official=False)
    rows = []
    for index in range(len(dataset)):
        rows.extend(check_graph(solver, dataset, index, args))
    summary = {}
    for chunk in CHUNKS:
        summary[f"chunk{chunk}"] = dict(
            decode_ms=round(median(r[f"chunk{chunk}"]["decode_ms"] for r in rows), 3),
            capture_ms=round(median(r[f"chunk{chunk}"]["capture_ms"] for r in rows), 3),
            speedup=round(median(r[f"chunk{chunk}"]["speedup"] for r in rows), 2))
    summary["eager_ms"] = round(median(r["eager_ms"] for r in rows), 3)
    best = max(CHUNKS, key=lambda c: summary[f"chunk{c}"]["speedup"])
    summary["best_chunk"] = best
    print("identical bitsets on "
          f"{len(rows)} states; eager {summary['eager_ms']} ms; "
          + " ".join(f"chunk{c}: {summary[f'chunk{c}']['decode_ms']} ms "
                     f"({summary[f'chunk{c}']['speedup']}x)" for c in CHUNKS)
          + f"; best chunk {best}", flush=True)
    payload = dict(version="v8_positioning", mode="check_decode", summary=summary, rows=rows,
                   config=vars(args), **provenance(args.reference_seed))
    output = Path(args.result_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"check_decode_ref{args.reference_seed}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
