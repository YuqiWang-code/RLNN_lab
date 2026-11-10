"""Apply the P0 gate and graph-paired validation gates for SC-context."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np


ARMS = ("baseline", "v7_mlp", "v7_message", "v7_mamba", "v7_shuffle_mamba")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def p0(root, width):
    stage = root / "preflight"
    correctness = all(read(stage / name / "run.json")["status"] == "completed"
                      for name in ("cpu", "cuda_no", "cuda_fp16", "permutation"))
    speed = {precision: read(stage / f"profile_{width}_{precision}" / "run.json")["ratio"]
             for precision in ("no", "fp16")}
    info = read(stage / "information_probe" / "run.json")
    ratios = max(speed.values())
    repeat = 1.08 < ratios <= 1.12
    stable_direction = info["information_direction"]
    reliability = info["mean_label_split_spearman"]
    if not correctness:
        status = "fix_correctness"
    elif repeat:
        status = "remeasure"
    elif ratios > 1.12:
        status = "profile_width16" if width != 16 else "no_go_speed"
    elif stable_direction:
        status = "go_training"
    elif reliability < .25:
        status = "small_pilot_only"
    else:
        status = "no_go_information"
    return dict(status=status, correctness=correctness, width=width, speed_ratios=speed,
                proxy_auc=info["mean_metrics"], proxy_deltas=info["paired_delta_auc"],
                proxy_reliability=reliability, weak_proxy_threshold=.25)


def evaluate(root):
    def collect(protocol, arm, post):
        pairs, runs = [], []
        for ref in range(3):
            folder = root / f"{protocol}_{arm}_ref{ref}" / post
            metadata = read(folder / "run.json")
            if metadata["split"] != "validation" or metadata["samples"] != 100:
                raise ValueError(f"Expected the 100 validation graphs in {folder}")
            with (folder / "per_graph.csv").open(newline="", encoding="utf-8") as stream:
                pairs.append({r["graph_id"]: int(r["final_size"]) for r in csv.DictReader(stream)})
            runs.append(metadata)
        return pairs, runs

    protocols = (("fixed", "none"), ("fixed", "pave"), ("budget05", "pave"), ("budget10", "pave"))
    data = {(protocol, post, arm): collect(protocol, arm, post)
            for protocol, post in protocols for arm in ARMS}
    reports, effects = [], []
    random = np.random.default_rng(7307)
    intervals = random.integers(0, 100, (10000, 100))

    def paired(protocol, post, arm, control):
        a, _ = data[protocol, post, arm]
        b, _ = data[protocol, post, control]
        gids = list(a[0])
        if any(set(gids) != values.keys() for values in a+b):
            raise ValueError("Graph IDs differ between seeds or methods")
        delta = np.asarray([[f[gid]-g[gid] for gid in gids] for f, g in zip(a, b)])
        by_graph = delta.mean(axis=0)
        ci = np.quantile(by_graph[intervals].mean(axis=1), [.025, .975])
        return dict(protocol=protocol, post=post, arm=arm, control=control, delta=float(delta.mean()),
                    seed_deltas=delta.mean(axis=1).tolist(), ci95=ci.tolist(),
                    wins=int((delta>0).sum()), ties=int((delta==0).sum()), losses=int((delta<0).sum()))

    for protocol, post in protocols:
        for arm in ARMS:
            _, runs = data[protocol, post, arm]
            sizes = [r["mean_size"] for r in runs]
            effect = paired(protocol, post, arm, "baseline")
            reports.append(dict(protocol=protocol, post=post, arm=arm, mean_size=mean(sizes), sd=stdev(sizes),
                                delta_size=effect["delta"], feasible_rate=mean(r["feasible_rate"] for r in runs),
                                mean_solve_s=mean(r["mean_solve_time_s"] for r in runs),
                                mean_search_steps=mean(r["mean_search_steps"] for r in runs),
                                params=runs[0]["params"], peak_memory_mib=max(r["peak_memory_mib"] for r in runs)))
            if arm != "baseline":
                effects.append(effect)
    mamba = "v7_mamba"
    def scores(protocol, post, arm, metric="mean_size"):
        return [r[metric] for r in data[protocol, post, arm][1]]
    def passes(protocol, post, arm, minimum, *, each=False):
        a, b = scores(protocol, post, arm), scores(protocol, post, "baseline")
        return all(x > max(minimum, y) for x, y in zip(a, b)) if each else mean(a) > max(minimum, mean(b))
    main_effect = [paired("fixed", "pave", mamba, control) for control in ("v7_mlp", "v7_message")]
    speed = [read(root / f"trained_profile_ref0_{p}" / "run.json")["ratio"] for p in ("no", "fp16")]
    stability = read(root / "trained_permutation_ref0" / "run.json")["summary"]
    gates = dict(
        feasibility=all(r["feasible_rate"] == 1 for r in reports),
        bare=passes("fixed", "none", mamba, 19.62),
        pave=passes("fixed", "pave", mamba, 19.69, each=True),
        budget05=passes("budget05", "pave", mamba, 19.70),
        budget10=passes("budget10", "pave", mamba, 19.85),
        steps10=mean(scores("budget10", "pave", mamba, "mean_search_steps")) >= 250,
        mamba_specific=all(min(row["seed_deltas"]) > 0 and row["ci95"][0] > 0 for row in main_effect),
        speed=max(speed) <= 1.08,
        stability=all(row["context_delta_pave"] >= row["baseline_delta_pave"] - .05 for row in stability.values()),
    )
    controls = [arm for arm in ("v7_mlp", "v7_message") if all(
        passes(protocol, "pave", arm, threshold) for protocol, threshold in (("budget05", 19.70), ("budget10", 19.85)))]
    status = "go_a" if all(gates.values()) else "go_b" if controls and not gates["mamba_specific"] else "no_go"
    return dict(status=status, gates=gates, controls=controls, summary=reports, paired=effects+main_effect)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/v7_context"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("p0", "validation"), required=True)
    parser.add_argument("--width", type=int, default=32)
    args = parser.parse_args()
    result = p0(args.root, args.width) if args.stage == "p0" else evaluate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result if args.stage == "p0" else dict(status=result["status"], gates=result["gates"]), indent=2))
