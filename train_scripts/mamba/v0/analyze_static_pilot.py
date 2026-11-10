"""Reanalyze P0 and summarize graph-paired static-prior MIS results."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3] / "outputs/v7_context"
ARMS = ("baseline", "mlp", "message")
PROTOCOLS = (("fixed", "none"), ("fixed", "pave"), ("budget05", "pave"), ("budget10", "pave"))


def load(folder):
    return json.loads((folder / "run.json").read_text(encoding="utf-8"))


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def index(records, *keys):
    pairs = {tuple(row[k] for k in keys): row for row in records}
    if len(pairs) != len(records):
        raise ValueError(f"Duplicate key for {keys}")
    return pairs


def effect(delta):
    delta = np.asarray(delta, dtype=float)
    draws = np.random.default_rng(7323).integers(len(delta), size=(10000, len(delta)))
    loo = (delta.sum() - delta) / (len(delta) - 1)
    return dict(mean=float(delta.mean()), median=float(np.median(delta)),
                ci95=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                wins=int((delta > 0).sum()), ties=int((delta == 0).sum()), losses=int((delta < 0).sum()),
                worst=float(delta.min()), best=float(delta.max()), loo_range=[float(loo.min()), float(loo.max())])


def pairs(candidate, reference, metric, multiplier=1):
    if candidate.keys() != reference.keys():
        raise ValueError(f"Graph pairs differ for {metric}")
    return np.array([(float(candidate[g][metric]) - float(reference[g][metric])) * multiplier for g in reference])


def preflight(stage):
    profiles = {}
    for folder in sorted(stage.glob("profile_*")):
        data = rows(folder / "per_graph.csv")
        grouped = {arm: index([r for r in data if r["arm"] == arm], "graph_id", "repeat")
                   for arm in ("baseline", "mamba")}
        by_graph = defaultdict(list)
        for graph, repeat in grouped["baseline"]:
            by_graph[graph].append(repeat)
        if any(len(repeats) != 5 for repeats in by_graph.values()) or len(by_graph) != 20:
            raise ValueError(f"Unexpected timing sample in {folder}")
        fields = ("bare_s", "prepare_capture_s", "rollout_s", "a3_s", "pave_s", "full_s")
        a, b = grouped["mamba"], grouped["baseline"]
        graph_rows = []
        for graph, repeats in by_graph.items():
            keys = [(graph, rep) for rep in repeats]
            row = {field: 1000 * float(np.mean([float(a[k][field])-float(b[k][field]) for k in keys]))
                   for field in fields}
            means = {name: float(np.mean([float(records[k]["bare_s"]) for k in keys]))
                     for name, records in grouped.items()}
            row.update(graph_id=graph, ratio=means["mamba"]/means["baseline"],
                       static_ms=1000*float(np.mean([float(a[k]["static_features_s"]) for k in keys])),
                       encode_ms=1000*float(np.mean([float(a[k]["context_s"]) for k in keys])))
            graph_rows.append(row)
        profiles[folder.name] = dict(
            ratio=load(folder)["ratio"], graph_count=len(graph_rows), repeats=5,
            components_ms={field: effect([r[field] for r in graph_rows]) for field in fields},
            graph_ratio_p95=float(np.quantile([r["ratio"] for r in graph_rows], .95)),
            graphs_over_gate=sum(r["ratio"] > 1.08 for r in graph_rows),
            matched_100step_size=sum(a[k]["base_size"] == b[k]["base_size"] for k in b),
            per_graph=graph_rows)
    data = rows(stage / "information_probe/per_graph.csv")
    by_arm = {arm: index([r for r in data if r["arm"] == arm], "graph_id")
              for arm in ("local_mlp", "full_mlp", "message", "mamba")}
    proxy = {}
    for arm, control in (("full_mlp", "local_mlp"), ("message", "local_mlp"), ("message", "full_mlp"),
                         ("mamba", "local_mlp"), ("mamba", "full_mlp"), ("mamba", "message")):
        proxy[f"{arm}_vs_{control}"] = {
            metric: effect(pairs(by_arm[arm], by_arm[control], metric))
            for metric in ("auc", "spearman", "brier")
        }
    data = rows(stage / "budget_16_optimized_no/per_graph.csv")
    steps = {}
    for budget in ("0.5", "1.0"):
        a, b = (index([r for r in data if r["budget"] == budget and r["arm"] == arm], "graph_id")
                for arm in ("mamba", "baseline"))
        steps[budget] = effect(pairs(a, b, "search_steps"))
    return dict(profiles=profiles, proxy=proxy, p0_budget_step_difference=steps)


def pilot(path):
    summary, paired, transitions, training = [], [], [], []
    reference_keys = None
    repeated = (path / "repeat_completed.txt").is_file()
    protocols = PROTOCOLS + (("repeat05", "pave"), ("repeat10", "pave")) if repeated else PROTOCOLS
    budget_rows = {}
    for protocol, post in protocols:
        methods, metadata = {}, {}
        for arm in ARMS:
            directory = path / f"{protocol}_{arm}" / post
            methods[arm] = index(rows(directory / "per_graph.csv"), "graph_id", "search_seed")
            metadata[arm] = load(directory)
            if reference_keys is None:
                reference_keys = methods[arm].keys()
            if methods[arm].keys() != reference_keys or len(reference_keys) != 50:
                raise ValueError(f"Different graph cohort or seed at {directory}")
            for key in ("backbone_weight_sha256", "pave_sha256", "cuda_graph", "split", "gpu"):
                if metadata[arm][key] != metadata["baseline"][key]:
                    raise ValueError(f"{key} mismatch for {directory}")
            record = metadata[arm]
            data = list(methods[arm].values())
            base = np.array([int(r["base_size"]) for r in data])
            final = np.array([int(r["final_size"]) for r in data])
            search = np.array([float(r["search_time_s"]) for r in data])
            summary.append(dict(
                protocol=protocol, post=post, arm=arm, graphs=len(final),
                mean_size=float(final.mean()), graph_sd=float(final.std(ddof=1)),
                mean_base_size=float(base.mean()), mean_pave_gain=float((final-base).mean()),
                pave_improved=int((final>base).sum()), mean_solve_s=record["mean_solve_time_s"],
                p95_solve_s=record["p95_solve_s"], mean_search_s=float(search.mean()),
                mean_post_s=record["mean_post_time_s"], mean_steps=record["mean_search_steps"],
                feasible=record["feasible_rate"], params=record["params"],
                peak_mib=record["peak_memory_mib"], overruns=record["deadline_overruns"],
                fallbacks=record["fallback_count"], gpu=record["gpu"]))
        if protocol != "fixed":
            budget_rows[protocol] = methods
        for candidate, control in (("mlp", "baseline"), ("message", "baseline"), ("message", "mlp")):
            a, b = methods[candidate], methods[control]
            delta = pairs(a, b, "final_size")
            pre = pairs(a, b, "base_size")
            paired.append(dict(protocol=protocol, post=post, arm=candidate, control=control,
                               size=effect(delta), base_size=effect(pre), pave_gain=effect(delta-pre),
                               solve_ms=effect(pairs(a, b, "solve_time_s", 1000)),
                               search_ms=effect(pairs(a, b, "search_time_s", 1000)),
                               steps=effect(pairs(a, b, "search_steps"))))
            if post == "pave":
                transitions.append(dict(protocol=protocol, arm=candidate, control=control,
                                        pre_win_lost=int(((pre>0)&(delta<=0)).sum()),
                                        pre_loss_recovered=int(((pre<0)&(delta>=0)).sum()),
                                        pre_tie_became_win=int(((pre==0)&(delta>0)).sum()),
                                        pre_tie_became_loss=int(((pre==0)&(delta<0)).sum())))
    for arm in ARMS[1:]:
        folder = path / f"train_{arm}"
        data = load(folder)
        history = rows(folder / "training.csv")[-1]
        training.append(dict(arm=arm, updates=data["updates"], wall_s=data["train_wall_s"],
                             peak_mib=data["peak_memory_mib"], alpha=float(history["alpha"]),
                             delta_logit_rms=float(history["delta_logit_rms"]),
                             delta_prob_mean_abs=float(history["delta_prob_mean_abs"]),
                             train_flips=float(history["train_flips"])))
    budget_replication = []
    if repeated:
        for suffix in ("05", "10"):
            first, second = (budget_rows[f"{prefix}{suffix}"] for prefix in ("budget", "repeat"))
            for arm in ARMS:
                base = pairs(second[arm], first[arm], "final_size")
                base_steps = pairs(second[arm], first[arm], "search_steps")
                entry = dict(budget="0.5" if suffix == "05" else "1.0", arm=arm,
                             repeat_minus_first_size=effect(base), repeat_minus_first_steps=effect(base_steps))
                if arm != "baseline":
                    first_size = pairs(first[arm], first["baseline"], "final_size")
                    second_size = pairs(second[arm], second["baseline"], "final_size")
                    first_steps = pairs(first[arm], first["baseline"], "search_steps")
                    second_steps = pairs(second[arm], second["baseline"], "search_steps")
                    entry.update(pooled_size_vs_b=effect((first_size+second_size)/2),
                                 pooled_steps_vs_b=effect((first_steps+second_steps)/2),
                                 won_both_runs=int(((first_size>0)&(second_size>0)).sum()),
                                 lost_both_runs=int(((first_size<0)&(second_size<0)).sum()))
                budget_replication.append(entry)
    return dict(summary=summary, paired=paired, transitions=transitions, training=training,
                budget_replication=budget_replication)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--p0-only", action="store_true")
    args = parser.parse_args()
    destination = args.root / "analysis/static_prior_pilot"
    destination.mkdir(parents=True, exist_ok=True)
    data = {"p0_reanalysis": preflight(args.root / "preflight")}
    if not args.p0_only:
        data["pilot"] = pilot(args.root / "static_prior_pilot")
    for name, record in data.items():
        (destination / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
