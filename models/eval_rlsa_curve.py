"""RLSA size-versus-clock curve on the author test split.

Why this exists
---------------
The paper reports one RLSA (=KLSA) point: 19.97 in 35 s over the whole test set.
Every "we are faster / better than RLSA" sentence in the v8 write-up leans on a
*single* clock, which is the easiest thing for a reviewer to reject: RLSA is a
pure simulated annealer with no training, so the interesting question is not its
quality at 0.070 s but its whole quality-versus-time curve.  If RLSA at 1.0 s is
already above our 1.0 s number, the "best at 1 s" claim is dead and we should not
write it.  If RLSA at 1.0 s is below us, the claim becomes a measured fact.

Nothing here trains or tunes anything: RLSA is run exactly as
``solvers.RLSA.evaluate_single`` defines it (kth-value acceptance, geometric
cooling from ``tau0`` to 0), with its own A.3 decode, at the same seeds and on
the same graphs as ``eval_v8_padded``.  Each ``--num_ts`` entry is an
independent run with that schedule length, not a checkpoint of one long run,
because ``num_t`` also sets the cooling rate.

    CUDA_VISIBLE_DEVICES=0 python -u models/eval_rlsa_curve.py \
        --split test --test_path models/data/MIS_tensors/RB_small_author_test \
        --test_samples 1000 --num_ts 50 100 200 500 1000 2000 4000 \
        --seeds 1017 --result_dir outputs/v8_macro/rlsa_curve
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch

from eval_v2 import TEST_CACHE, TEST, dataset_at, sync
from main import build_parser, set_seed
from mis_v3_search import SearchGraph
from mis_v7_recombine import is_independent
from solvers import RLSA

ROOT = Path(__file__).resolve().parents[1]


def curve_parser():
    parser = build_parser()
    parser.set_defaults(method="RLSA", test_path=None, mixed_precision="no", seed=1017,
                        num_k=20, num_t=1000, num_l=5, num_h=128, num_d=30, num_workers=0,
                        do_train=False, beta=1.02, tau0=0.01, skip_decode=False)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--num_ts", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000, 4000],
                        help="independent schedule lengths (the cooling rate depends on it)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1017])
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--clock", type=float, default=1.0, help="reference clock for the summary")
    return parser


def bits_of(mask):
    """Bool node mask -> integer bitmask (the convention of mis_v7_recombine)."""
    return int(sum(1 << int(node) for node in mask.nonzero().flatten().tolist()))


@torch.no_grad()
def run_rlsa(solver, graph, num_t, seed):
    """One RLSA trajectory with ``num_t`` its own cooling schedule, then A.3 decode."""
    solver.num_t = num_t
    generator = torch.Generator(device=graph.b.device).manual_seed(seed)
    mask = solver.evaluate_single(graph, return_solution=True, generator=generator)
    return mask


def main():
    args = curve_parser().parse_args()
    if args.test_path is None:
        if args.split == "validation":
            args.test_path = str(ROOT / "models/data/MIS_tensors/RB_small_val")
        else:
            args.test_path = str(TEST_CACHE if (TEST_CACHE / "tensor_manifest.json").is_file() else TEST)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    solver = RLSA(args)
    solver.prepare()
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    indices = list(range(len(dataset)))[args.shard::args.shards]

    rows = []
    started = perf_counter()
    for position, index in enumerate(indices):
        graph = dataset[index]
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        graph = graph.to(device)
        for seed in args.seeds:
            neighbours = SearchGraph(int(graph.num_nodes), edges, seed=seed).neighbors
            for num_t in args.num_ts:
                sync(device)
                tick = perf_counter()
                mask = run_rlsa(solver, graph, num_t, seed)
                sync(device)
                seconds = perf_counter() - tick
                size = int(mask.sum())
                if not is_independent(bits_of(mask), neighbours):
                    raise ValueError(f"{graph_id}: RLSA decode is not independent")
                rows.append(dict(graph_id=graph_id, seed=seed, num_t=num_t, size=size,
                                 wall_time_s=round(seconds, 5)))
        if (position + 1) % 25 == 0 or position + 1 == len(indices):
            parts = []
            for num_t in args.num_ts:
                chosen = [row for row in rows if row["num_t"] == num_t]
                if chosen:
                    parts.append(f"T{num_t}:{mean(r['size'] for r in chosen):.3f}"
                                 f"@{mean(r['wall_time_s'] for r in chosen):.3f}s")
            print(f"{position + 1}/{len(indices)} " + " ".join(parts), flush=True)

    tag = "" if args.shards == 1 else f"_shard{args.shard}"
    path = args.result_dir / f"rlsa_curve{tag}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_num_t = {}
    for num_t in args.num_ts:
        chosen = [row for row in rows if row["num_t"] == num_t]
        if not chosen:
            continue
        walls = [row["wall_time_s"] for row in chosen]
        entry = dict(sizes=mean(row["size"] for row in chosen), runs=len(chosen),
                     wall_mean_s=mean(walls), wall_median_s=median(walls),
                     wall_max_s=max(walls))
        per_num_t[num_t] = entry
        print(f"num_t={num_t:<5} size {entry['sizes']:.4f} "
              f"wall mean {entry['wall_mean_s']:.4f} median {entry['wall_median_s']:.4f} "
              f"max {entry['wall_max_s']:.4f}", flush=True)

    under = [n for n in per_num_t if per_num_t[n]["wall_mean_s"] <= args.clock]
    best_under = max(under, key=lambda n: per_num_t[n]["sizes"]) if under else None
    summary = dict(split=args.split, samples=len(indices), num_ts=args.num_ts,
                   seeds=args.seeds, num_k=args.num_k, num_d=args.num_d, tau0=args.tau0,
                   mixed_precision=args.mixed_precision, clock_s=args.clock,
                   best_under_clock=best_under,
                   best_under_clock_size=per_num_t[best_under]["sizes"] if best_under else None,
                   per_num_t={str(k): v for k, v in per_num_t.items()},
                   wall_time_s=round(perf_counter() - started, 2))
    if args.shards > 1:
        summary["shard"] = dict(index=args.shard, count=args.shards)
    (args.result_dir / f"rlsa_curve_ref{args.seed}{tag}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    if best_under is not None:
        print(f"RLSA best at mean wall <= {args.clock:g}s: num_t={best_under} "
              f"size {per_num_t[best_under]['sizes']:.4f} "
              f"({per_num_t[best_under]['wall_mean_s']:.3f}s)")


if __name__ == "__main__":
    main()
