"""Summarize the within-instance ER-small budget replication."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from compare_mis_runs import compare, read_rows, summarize


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        table = csv.DictWriter(stream, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    protocol = json.loads((args.root / "protocol.json").read_text())
    cells, seeds = protocol["cells"], protocol["seeds"]
    positions = protocol["williams_base_order"]
    graphs = protocol["graphs"]
    summaries, contrasts = [], []
    for cell_id, (budget, arm) in enumerate(cells):
        file = args.root / str(budget) / arm / "per_graph.csv"
        records = read_rows(file)
        expected = {(f"idx_{i}", str(s), "0") for i in range(graphs) for s in seeds}
        if set(records) != expected:
            raise ValueError(f"Incomplete cohort in {file}")
        with file.open(newline="") as stream:
            counters, returns = Counter(), []
            for row in csv.DictReader(stream):
                block, period = int(row["block"]), int(row["period"])
                if (positions[period] + block) % len(cells) != cell_id or int(row["seed"]) != seeds[block % len(seeds)]:
                    raise ValueError(f"Wrong treatment order in {file}")
                counters[period] += 1
                returns.append(float(row["wall_ms"]) <= budget)
        if len(counters) != len(cells) or set(counters.values()) != {len(expected) // len(cells)}:
            raise ValueError(f"Unbalanced treatment positions in {file}")
        metrics = summarize(records, budget)
        summaries.append(dict(budget_ms=budget, method=arm, graphs=graphs, pairs=len(records),
                              mean_size=metrics["mean_size"], mean_wall_ms=metrics["mean_wall_ms"],
                              median_wall_ms=metrics["wall_p50_ms"], p95_wall_ms=metrics["wall_p95_ms"],
                              ready=metrics["ready_by_deadline"], returned_on_time=sum(returns),
                              mean_steps=metrics["mean_steps"]))
        if arm == "learned":
            for ref in ("degree", "online_mis"):
                other = args.root / str(budget) / ref / "per_graph.csv"
                delta = compare(file, other, deadline_ms=budget)
                contrasts.append(dict(budget_ms=budget, reference_budget_ms=budget, candidate=arm,
                                      reference=ref, mean_paired_delta=delta["mean_paired_delta"],
                                      ci95_low=delta["ci95"][0], ci95_high=delta["ci95"][1],
                                      wins=delta["wins"], ties=delta["ties"], losses=delta["losses"]))
    for arm in ("learned", "degree", "online_mis"):
        delta = compare(args.root / "100" / arm / "per_graph.csv", args.root / "50" / arm / "per_graph.csv",
                        deadline_ms=100)
        contrasts.append(dict(budget_ms=100, reference_budget_ms=50, candidate=arm, reference=arm,
                              mean_paired_delta=delta["mean_paired_delta"], ci95_low=delta["ci95"][0],
                              ci95_high=delta["ci95"][1], wins=delta["wins"], ties=delta["ties"],
                              losses=delta["losses"]))
    write_rows(args.root / "summary.csv", summaries)
    write_rows(args.root / "paired.csv", contrasts)


if __name__ == "__main__":
    main()
