"""Summarize the structure_v1 runs with graph-paired intervals and admission gates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "outputs/structure_v1"
ARMS = ("baseline", "degree", "gate", "gain", "all", "degree_out", "all_out")
PROTOCOLS = {"fixed": ("fixed_%s_ref%d", ("none", "pave")),
             "fixed1000": ("fixed1000_%s_ref%d", ("none", "pave")),
             "budget05": ("budget05_%s_ref%d", ("pave",)),
             "budget10": ("budget10_%s_ref%d", ("pave",))}


def load(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def effect(deltas, draws):
    values = np.asarray(deltas, dtype=float)
    return dict(delta=float(values.mean()),
                ci95=np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist(),
                wins=int((values > 0).sum()), ties=int((values == 0).sum()),
                losses=int((values < 0).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    results = []
    for protocol, (pattern, posts) in PROTOCOLS.items():
        for post in posts:
            tables = {}
            for arm in ARMS:
                folders = sorted(args.root.glob((pattern % (arm, args.ref)) + '*'))
                rows = {}
                for folder in folders:
                    source = folder / post / "per_graph.csv"
                    if not source.is_file():
                        continue
                    for row in load(source):
                        rows.setdefault(row["graph_id"], []).append(row)
                if rows:
                    tables[arm] = rows
            if "baseline" not in tables:
                continue
            base = tables["baseline"]
            graphs = sorted(base)
            draws = np.random.default_rng(8241).integers(len(graphs), size=(10000, len(graphs)))
            for arm, table in tables.items():
                if set(table) != set(graphs):
                    raise ValueError(f"{protocol}/{post}/{arm} does not cover the baseline graphs")
                size = [np.mean([int(row["final_size"]) for row in table[g]])
                        - np.mean([int(row["final_size"]) for row in base[g]]) for g in graphs]
                time = [np.mean([float(row["solve_time_s"]) for row in table[g]])
                        - np.mean([float(row["solve_time_s"]) for row in base[g]]) for g in graphs]
                results.append(dict(protocol=protocol, post=post, arm=arm, graphs=len(graphs),
                                    seeds=sorted({len(v) for v in table.values()}),
                                    mean_size=float(np.mean([np.mean([int(r["final_size"]) for r in table[g]])
                                                             for g in graphs])),
                                    base_mean=float(np.mean([np.mean([int(r["final_size"]) for r in base[g]])
                                                             for g in graphs])),
                                    mean_solve_s=float(np.mean([np.mean([float(r["solve_time_s"]) for r in table[g]])
                                                                for g in graphs])),
                                    base_solve_s=float(np.mean([np.mean([float(r["solve_time_s"]) for r in base[g]])
                                                                for g in graphs])),
                                    mean_steps=float(np.mean([np.mean([int(r["search_steps"]) for r in table[g]])
                                                              for g in graphs])),
                                    feasible=all(int(row["conflicts"]) == 0 for v in table.values() for row in v),
                                    size=effect(size, draws), time_ms=effect([1000 * value for value in time], draws)))

    def row(protocol, post, arm):
        for entry in results:
            if (entry["protocol"], entry["post"], entry["arm"]) == (protocol, post, arm):
                return entry
        return None

    gates = {}
    for arm in ARMS[1:]:
        checks = {}
        fixed = row("fixed", "pave", arm)
        checks["fixed"] = bool(fixed and fixed["size"]["ci95"][0] > 0)
        slow = row("fixed", "none", arm)
        checks["speed"] = bool(slow and slow["mean_solve_s"] <= 1.08 * slow["base_solve_s"])
        budgets = [row(name, "pave", arm) for name in ("budget05", "budget10")]
        checks["budget"] = all(entry and entry["size"]["ci95"][0] > 0 for entry in budgets)
        checks["feasible"] = all(entry and entry["feasible"] for entry in results if entry["arm"] == arm)
        gates[arm] = dict(checks=checks, advance=all(checks.values()))
    analysis = dict(ref=args.ref, results=results, gates=gates,
                    status="advance" if any(entry["advance"] for entry in gates.values()) else "no_go")

    target = args.root / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    (target / f"summary_ref{args.ref}.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    (target / f"decision_ref{args.ref}.json").write_text(
        json.dumps(dict(status=analysis["status"], gates=gates), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=analysis["status"], gates=gates), indent=2))


if __name__ == "__main__":
    main()
