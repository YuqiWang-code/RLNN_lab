"""Summarize the fused solver's load, search and decode timings by test split."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean

from summarize_scale_deadline import SPLITS, summarize


def timings(root, split):
    run = root / split / "fused"
    scores = summarize(run / "per_graph.csv", SPLITS[split])
    with (run / "per_graph.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    config = json.loads((run / "padded/padded_ref0.json").read_text(encoding="utf-8"))
    on_time_sizes = [float(r["final_size"]) for r in rows if r["final_ready_ms"]]
    misses = [r for r in rows if not r["final_ready_ms"]]
    return dict(
        split=split, graphs=config["graphs"], **scores, mean_steps=fmean(float(r["steps"]) for r in rows),
        mean_size_when_on_time=fmean(on_time_sizes) if on_time_sizes else None,
        mean_load_ms=fmean(float(r["load_ms"]) for r in rows),
        min_load_ms=min(float(r["load_ms"]) for r in rows),
        mean_decode_ms=fmean(float(r["decode_ms"]) for r in rows),
        loads_over_200_ms=sum(float(r["load_ms"]) > 200 for r in rows),
        runs_with_completed_decode=sum(bool(r["last_decode_ready_ms"]) for r in rows),
        misses_with_late_decode=sum(float(r["last_decode_ready_ms"] or 0) > 200 for r in misses),
        misses_without_search_steps=sum(int(r["steps"]) == 0 for r in misses),
        function_returns_after_200_ms=sum(int(r["deadline_overrun"]) for r in rows),
        engine_setup_s=config["setup_s"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    root = parser.parse_args().results
    stats = [timings(root, split) for split in ("RB_large", "ER_small", "ER_large")]
    with (root / "fused_timing.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)


if __name__ == "__main__":
    main()
