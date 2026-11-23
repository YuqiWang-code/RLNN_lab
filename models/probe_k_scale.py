"""Does the reported size grow with the number of particles at a *fixed* number of steps?

The deployed engine runs ``num_k=20`` particles (the training width) for the whole
one-second budget and reports the A.3 decode of the best archive entry.  RLSA, the
strongest solver on this benchmark, instead runs ``num_k=200`` particles for only
~300 steps and also reports the best of the population.

Because the metric is a *maximum over the population*, the two designs differ in
the quantity that matters: staying at ``num_k=20`` throws away the tail.  The
question this probe answers is empirical, not theoretical: at a fixed step count,
how does the best-of-K decoded size move as K grows, and how much wall time does
that cost?  The per-particle mean is reported next to the max so it is visible
whether the gain comes from a shifted distribution or from its tail.
"""
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v8_padded import build_parser, shape_of
from eval_v2 import TEST, TEST_CACHE, dataset_at, load_policy
from mis_v8_pad import PaddedSearch


def main():
    parser = build_parser()
    parser.add_argument("--num_ks", type=int, nargs="+", default=[20, 50, 200, 800])
    parser.add_argument("--blocks", type=int, default=70, help="blocks of 10 steps")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.test_path is None:
        args.test_path = str(TEST_CACHE if (TEST_CACHE / "tensor_manifest.json").is_file() else TEST)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    solver = load_policy(args)
    dataset = dataset_at(args.test_path, solver.dtype, args.samples, official=True)
    nodes, capacity = shape_of(dataset, len(dataset))
    print(f"shapes n_max={nodes} capacity={capacity}", flush=True)

    rows = []
    for num_k in args.num_ks:
        solver.num_k = num_k
        engine = PaddedSearch(solver, n_max=nodes, capacity=capacity, chunk=10, decode_chunk=4)
        for index in range(len(dataset)):
            graph = dataset[index]
            engine.load(graph, graph.edge_index, args.seed)
            engine.run(perf_counter() + 1e9, lambda states: None, max_blocks=args.blocks)
            torch.cuda.synchronize()
            started = perf_counter()
            solutions = engine.decoder.decode(engine.best_states)[:engine.nodes]
            torch.cuda.synchronize()
            decode_ms = 1e3 * (perf_counter() - started)
            sizes = solutions.sum(dim=0).to(torch.int64).cpu().tolist()
            rows.append(dict(graph_id=f"idx_{index}", num_k=num_k, steps=engine.chunk * args.blocks,
                             best=max(sizes), mean_size=round(mean(sizes), 3),
                             p90=sorted(sizes)[int(0.9 * (len(sizes) - 1))],
                             decode_ms=round(decode_ms, 2)))
        chosen = [row for row in rows if row["num_k"] == num_k]
        print(f"K={num_k:<4} steps {chosen[0]['steps']} best {mean(r['best'] for r in chosen):.4f} "
              f"mean {mean(r['mean_size'] for r in chosen):.3f} "
              f"p90 {mean(r['p90'] for r in chosen):.2f} decode {chosen[0]['decode_ms']:.1f}ms",
              flush=True)

    path = Path(args.result_dir) / "k_scale.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for num_k in args.num_ks:
        chosen = [row for row in rows if row["num_k"] == num_k]
        summary[str(num_k)] = dict(best=mean(r["best"] for r in chosen),
                                   mean_size=mean(r["mean_size"] for r in chosen),
                                   decode_ms=chosen[0]["decode_ms"])
    (Path(args.result_dir) / "k_scale.json").write_text(
        json.dumps(dict(steps=args.blocks * 10, samples=len(dataset), summary=summary),
                   indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
