"""Aggregate the matched v6 pilot, speed probes and branch diagnostics."""

import csv
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/v6"
SEEDS = (42, 123, 999)


def run(name):
    return json.loads((OUT / name / "run.json").read_text(encoding="utf-8"))


def sizes(name):
    with (OUT / name / "per_graph.csv").open(encoding="utf-8", newline="") as stream:
        return {row["graph_id"]: int(row["final_size"]) for row in csv.DictReader(stream)}


def main():
    per_seed, pairs, summary = [], [], []
    for post in ("none", "pave"):
        grouped = {"baseline": [], "full": []}
        for ref, seed in enumerate(SEEDS):
            base = f"pilot_fixed_baseline_ref{ref}/{post}"
            full = f"pilot_fixed_full_ref{ref}/{post}"
            b_size, f_size = sizes(base), sizes(full)
            if len(b_size) != 100 or b_size.keys() != f_size.keys():
                raise ValueError(f"Inconsistent graph pairs for {post}, seed {seed}")
            diff = [f_size[graph] - b_size[graph] for graph in b_size]
            pairs.append(dict(post=post, seed=seed, graphs=len(diff), delta_size=mean(diff),
                              wins=sum(x > 0 for x in diff), ties=diff.count(0), losses=sum(x < 0 for x in diff)))
            for arm, name in (("baseline", base), ("full", full)):
                data = run(name)
                row = dict(arm=arm, post=post, seed=seed, reference_seed=ref, graphs=data["samples"],
                           mean_size=data["mean_size"], graph_sd=data["sd_graph_size"],
                           delta_size=mean(diff) if arm == "full" else 0.0,
                           mean_solve_s=data["mean_solve_time_s"], feasible_rate=data["feasible_rate"],
                           params=data["params"], peak_memory_mib=data["peak_memory_mib"], source=name)
                grouped[arm].append(row)
                per_seed.append(row)
        for arm, rows in grouped.items():
            means = [row["mean_size"] for row in rows]
            summary.append(dict(arm=arm, post=post, seeds=len(rows), graphs=100, mean_size=mean(means),
                                size_seed_sd=stdev(means), delta_size=mean(row["delta_size"] for row in rows),
                                mean_solve_s=mean(row["mean_solve_s"] for row in rows),
                                feasible_rate=mean(row["feasible_rate"] for row in rows),
                                params=rows[0]["params"],
                                peak_memory_mib=max(row["peak_memory_mib"] for row in rows)))

    profiles = []
    for method in ("eager", "captured", "fused", "pilot"):
        for precision in ("fp16", "no"):
            name = f"profile_{method}_{precision}"
            data = run(name)
            profiles.append(dict(method=method, precision="fp32" if precision == "no" else precision,
                                 graphs=data["samples"], repeats=data["repeats"], b_s=data["mean_search_s"]["B"],
                                 full_s=data["mean_search_s"]["SC-RLNN"], ratio=data["ratio"], speed_pass=data["pass_speed"],
                                 components=data.get("components"), source=name))

    health = []
    for ref, seed in enumerate(SEEDS):
        for arm in ("baseline", "full"):
            name = f"diagnose_{arm}_ref{ref}"
            data = run(name)
            health.append(dict(arm=arm, seed=seed, graphs=data["samples"], **data["summary"], source=name))

    train = [run(f"pilot_full_ref{ref}") for ref in range(3)]
    full_pave = [row for row in per_seed if row["arm"] == "full" and row["post"] == "pave"]
    full_bare = next(row for row in summary if row["arm"] == "full" and row["post"] == "none")
    full_health = [row for row in health if row["arm"] == "full"]
    checks = dict(
        speed=all(row["speed_pass"] for row in profiles if row["method"] == "pilot"),
        bare_size=full_bare["mean_size"] > 19.62,
        pave_all_seeds=all(row["mean_size"] > 19.69 and row["delta_size"] > 0 for row in full_pave),
        high_energy=all(row["high_logit_energy"] > 0.01 for row in full_health),
        deep_dirichlet=all(row["layer5_dirichlet_ratio"] > 0.25 for row in full_health),
    )
    destination = OUT / "analysis"
    destination.mkdir(exist_ok=True)
    record = dict(status="pilot_pass" if all(checks.values()) else "no_go", gates=checks,
                  summary=summary, per_seed=per_seed, paired=pairs, profiles=profiles, health=health,
                  train=[{key: row[key] for key in ("seed", "completed_epochs", "updates", "best_validation_size",
                                                   "peak_memory_mib", "train_wall_s")} for row in train])
    (destination / "pilot.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=record["status"], gates=checks, summary=summary,
                          train=record["train"]), indent=2))


if __name__ == "__main__":
    main()
