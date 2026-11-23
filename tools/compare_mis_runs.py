"""Summarize paired MIS runs from their per-graph records."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

import numpy as np


PROTOCOL_FIELDS = (
    "cell", "budget", "fixed_steps", "search_backend", "compile_model", "dtype", "dense",
    "decode_schedule", "decode_backend", "decode_particles", "decode_every", "num_k",
    "reserve_ms", "skip_incumbent", "no_pave",
)


def read_rows(path, pattern="**/v11_rows.csv"):
    files = [path] if path.is_file() else sorted(path.glob(pattern))
    if not files:
        raise ValueError(f"{path}: no per-graph CSV")
    rows = {}
    protocol = set()
    for file in files:
        with file.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["graph_id"], row.get("seed"), row.get("repeat", "0"))
                if key in rows:
                    raise ValueError(f"{file}: duplicate graph/seed/repeat {key}")
                protocol.add(tuple(row.get(field) for field in PROTOCOL_FIELDS))
                size = float(row["final_size"] if "final_size" in row else row["independent_set_size"])
                wall = (float(row["wall_ms"]) if "wall_ms" in row else 1000 * float(row["solve_time_s"]))
                ready = row.get("final_ready_ms")
                ready = wall if ready is None else (float(ready) if ready else None)
                steps = row.get("steps")
                rows[key] = (size, wall, ready, int(steps) if steps else None)
    if not rows or len(protocol) != 1:
        raise ValueError(f"{path}: expected one evaluation configuration")
    return rows


def summarize(rows, deadline_ms):
    sizes, walls, readiness, steps = zip(*rows.values())
    return dict(
        source_rows=len(rows),
        mean_size=fmean(sizes),
        mean_wall_ms=fmean(walls),
        wall_p50_ms=float(np.quantile(walls, 0.50)),
        wall_p95_ms=float(np.quantile(walls, 0.95)),
        mean_steps=fmean(steps) if None not in steps else None,
        ready_by_deadline=sum(t is not None and t <= deadline_ms for t in readiness),
    )


def compare(candidate_path, reference_path, deadline_ms=200, candidate_glob="**/v11_rows.csv"):
    candidate = read_rows(candidate_path, candidate_glob)
    reference = read_rows(reference_path)
    if {key[0] for key in candidate} != {key[0] for key in reference}:
        raise ValueError("Candidate and reference contain different graph IDs")
    fixed_reference = all(seed is None for _, seed, _ in reference)
    if not fixed_reference and candidate.keys() != reference.keys():
        raise ValueError("Candidate and reference contain different graph/seed/repeat pairs")
    per_graph = {}
    for key, (size, _, _, _) in candidate.items():
        graph_id, _, _ = key
        reference_key = (graph_id, None, "0") if fixed_reference else key
        per_graph.setdefault(graph_id, []).append(size - reference[reference_key][0])
    effects = np.array([fmean(differences) for differences in per_graph.values()])
    rng = np.random.default_rng(42)
    samples = np.concatenate([
        effects[rng.integers(0, len(effects), size=(500, len(effects)))].mean(axis=1)
        for _ in range(10)
    ])
    deltas = [d for ds in per_graph.values() for d in ds]
    return dict(
        candidate=summarize(candidate, deadline_ms),
        reference=summarize(reference, deadline_ms),
        deadline_ms=deadline_ms,
        graph_count=len(effects),
        pairs=len(deltas),
        mean_paired_delta=float(effects.mean()),
        ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
        wins=sum(d > 0 for d in deltas),
        ties=sum(d == 0 for d in deltas),
        losses=sum(d < 0 for d in deltas),
        same_seed_protocol=not fixed_reference,
        sources=dict(candidate=candidate_path.as_posix(), candidate_glob=candidate_glob,
                     reference=reference_path.as_posix()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--candidate-glob", default="**/v11_rows.csv",
                        help="select one arm from a candidate directory")
    parser.add_argument("--deadline-ms", type=float, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(
        compare(args.candidate, args.reference, args.deadline_ms, args.candidate_glob),
        indent=2,
    ) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
