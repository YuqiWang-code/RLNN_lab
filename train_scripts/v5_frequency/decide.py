"""Summarize v5 evaluations and apply the preregistered quality and budget gates."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/v5_frequency"


def write(path, records):
    if records:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def load(root):
    runs, graphs = [], {}
    for path in sorted(root.glob("eval_*/*/run.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["status"] != "completed" or record["split"] != "validation":
            continue
        cfg = record["config"]
        key = (record["protocol"], cfg["mixed_precision"], record["post"],
               cfg.get("time_budget_s"), record["method"])
        seed = record["train_seed"]
        with path.with_name("per_graph.csv").open(encoding="utf-8", newline="") as stream:
            rows = {row["graph_id"]: int(row["final_size"]) for row in csv.DictReader(stream)}
        if len(rows) != record["samples"] or (key, seed) in graphs:
            raise ValueError(f"Duplicate or incomplete evaluation: {path}")
        graphs[key, seed] = rows
        runs.append(dict(protocol=key[0], precision=key[1], post=key[2], budget_s=key[3],
                         method=key[4], seed=seed, graphs=len(rows), mean_size=record["mean_size"],
                         mean_solve_s=record["mean_solve_time_s"], p95_s=record["p95_time_s"],
                         feasible_rate=record["feasible_rate"], params=record["params"],
                         peak_memory_mib=record["peak_memory_mib"], mean_steps=record["mean_search_steps"],
                         mean_post_s=record["mean_post_time_s"], deadline_overruns=record["deadline_overruns"],
                         max_overrun_s=record["max_deadline_overrun_s"]))
    return runs, graphs


def compare(graphs, context, candidate, control):
    seeds = [seed for seed in range(3) if (context + (candidate,), seed) in graphs and
             (context + (control,), seed) in graphs]
    if not seeds:
        return None
    columns = []
    for seed in seeds:
        a, b = (graphs[context + (method,), seed] for method in (candidate, control))
        if a.keys() != b.keys():
            raise ValueError(f"Graph sets differ for {candidate}/{control}/seed{seed}")
        columns.append([a[gid]-b[gid] for gid in sorted(a)])
    delta = np.asarray(columns).T
    by_graph = delta.mean(1)
    rng = np.random.default_rng(20261109)
    draws = rng.choice(by_graph, size=(4000, len(by_graph)), replace=True).mean(1)
    return dict(protocol=context[0], precision=context[1], post=context[2], budget_s=context[3],
                candidate=candidate, control=control, seed_count=len(seeds), graph_count=len(by_graph),
                mean_delta=float(delta.mean()), ci_low=float(np.quantile(draws, .025)), ci_high=float(np.quantile(draws, .975)),
                positive_seeds=int((delta.mean(0) > 0).sum()), seed_deltas=json.dumps(dict(zip(seeds, delta.mean(0).tolist()))),
                wins=int((delta > 0).sum()), ties=int((delta == 0).sum()), losses=int((delta < 0).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    root = args.root
    runs, graphs = load(root)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    cohorts = {}
    for run in runs:
        key = tuple(run[k] for k in ("protocol", "precision", "post", "budget_s", "method"))
        cohorts.setdefault(key, []).append(run)
    summary = []
    for key, group in cohorts.items():
        sizes = [row["mean_size"] for row in group]
        summary.append(dict(zip(("protocol", "precision", "post", "budget_s", "method"), key),
                            seed_count=len(group), mean_size=mean(sizes), size_seed_sd=stdev(sizes) if len(sizes) > 1 else None,
                            **{name: mean(row[name] for row in group) for name in
                               ("mean_solve_s", "p95_s", "feasible_rate", "params", "mean_steps", "mean_post_s")},
                            peak_memory_mib=max(row["peak_memory_mib"] for row in group),
                            deadline_overruns=sum(row["deadline_overruns"] for row in group),
                            max_overrun_s=max(row["max_overrun_s"] for row in group)))
    pairs = []
    for context in sorted({key[:4] for key in cohorts}, key=str):
        for candidate in ("v5a", "v5a_conflict"):
            for control in ("baseline", "none", "analytic", "v5a_mlp", "v5a_message", "baseline_extended",
                            "baseline_rule23", "v5a_mlp_conflict", "v5a_message_conflict"):
                effect = compare(graphs, context, candidate, control)
                if effect:
                    pairs.append(effect)
    quality = [compare(graphs, ("fixed_work", "no", "none", None), "v5a", control)
               for control in ("baseline", "v5a_mlp", "v5a_message")]
    if any(e is None or e["seed_count"] != 3 or e["graph_count"] != 100 for e in quality):
        raise ValueError("The primary gate requires all three seeds and 100 paired graphs")
    quality_pass = all(e["mean_delta"] > 0 and e["positive_seeds"] >= 2 and e["ci_low"] > 0 for e in quality)
    budget = [compare(graphs, ("fixed_time", "fp16", "pave", .5), "v5a", control)
              for control in ("baseline", "v5a_mlp")]
    budget_complete = all(e is not None and e["seed_count"] == 3 and e["graph_count"] == 100 for e in budget)
    budget_pass = budget_complete and all(e["mean_delta"] >= 0 for e in budget)
    legal = all(row["feasible_rate"] == 1 for row in runs)
    result = dict(decision=("go_to_s2b" if quality_pass and budget_pass and legal else "no_go")
                  if budget_complete else "fixed_work_complete", quality_pass=quality_pass,
                  budget_complete=budget_complete, budget_pass=budget_pass, all_feasible=legal,
                  quality=quality, budget=budget, runs=len(runs))
    write(out / "per_seed.csv", runs)
    write(out / "summary.csv", summary)
    write(out / "paired_effects.csv", pairs)
    (out / "decision.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
