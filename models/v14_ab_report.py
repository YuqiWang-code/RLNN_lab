"""t21 / P0.5: three-arm table for the 0.2 s budget (C / A / B), three seeds.

C = leading decode on  + PAVE on   (outputs/v11/t7/a02_s{seed}_1000, t7 art)
A = leading decode off + PAVE on   (outputs/v11/t7/v13_noprelude_02s_s{seed})
B = leading decode off + PAVE off  (outputs/v11/t7/v13_nopave_02s_s{seed})

Outputs one JSON (means, paired bootstrap, per-graph drift) and a markdown
table on stdout.  Read-only: nothing is written outside ``--out``.
"""

import json
import statistics
from pathlib import Path

SEEDS = (1017, 1019, 1021)
ROOT = Path("outputs/v11/t7")
ARMS = {
    "C": "a02_s{seed}_1000",
    "A": "v13_noprelude_02s_s{seed}",
    "B": "v13_nopave_02s_s{seed}",
}
COLUMNS = ("search_size", "final_size", "steps", "wall_ms", "search_ms",
           "ms_per_step", "load_ms", "pave_gain", "deadline_overrun",
           "dtype", "dense", "fp32_csr", "skip_incumbent", "no_pave")


def read_rows(path):
    import csv
    with path.open() as handle:
        return [dict(entry) for entry in csv.DictReader(handle)]


def numeric(entry, key):
    try:
        return float(entry.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def load_arm(name):
    rows = {}
    for seed in SEEDS:
        path = ROOT / ARMS[name].format(seed=seed) / "v11_rows.csv"
        if not path.exists():
            raise SystemExit(f"missing {path} (arm {name} seed {seed})")
        for entry in read_rows(path):
            rows[(seed, entry["graph_id"])] = entry
    return rows


def check_tags(arm, rows):
    # arm C predates the two new flags (23 columns), so a missing key means "flag off"
    tags = {key: {(entry.get(key) or "0") for entry in rows.values()}
            for key in ("dtype", "dense", "fp32_csr", "skip_incumbent", "no_pave")}
    wanted = {
        "C": {"dtype": {"fp16"}, "dense": {"1"}, "fp32_csr": {"0"},
              "skip_incumbent": {"0"}, "no_pave": {"0"}},
        "A": {"dtype": {"fp16"}, "dense": {"1"}, "fp32_csr": {"0"},
              "skip_incumbent": {"1"}, "no_pave": {"0"}},
        "B": {"dtype": {"fp16"}, "dense": {"1"}, "fp32_csr": {"0"},
              "skip_incumbent": {"1"}, "no_pave": {"1"}},
    }[arm]
    for key, allowed in wanted.items():
        if tags[key] != allowed:
            raise SystemExit(f"arm {arm}: {key}={sorted(tags[key])} expected {sorted(allowed)}")
    return {key: sorted(values) for key, values in tags.items()}


def describe(rows, key):
    values = [numeric(entry, key) for entry in rows.values()]
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "n": len(values)}


def bootstrap(diffs, resamples=5000, seed=0):
    import torch
    tensor = torch.as_tensor(diffs, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(tensor.numel(), (resamples, tensor.numel()), generator=generator)
    means = tensor[index].mean(dim=1)
    return (float(tensor.mean()), float(torch.quantile(means, 0.025)),
            float(torch.quantile(means, 0.975)))


def paired(left, right, key):
    keys = sorted(set(left) & set(right))
    diffs = [numeric(left[k], key) - numeric(right[k], key) for k in keys]
    mean, low, high = bootstrap(diffs)
    wins = sum(1 for diff in diffs if diff > 0)
    losses = sum(1 for diff in diffs if diff < 0)
    return {"pairs": len(diffs), "mean": mean, "ci": [low, high],
            "wins": wins, "losses": losses, "ties": len(diffs) - wins - losses}


def drift(arm_rows, base_rows, key="search_size"):
    keys = sorted(set(arm_rows) & set(base_rows))
    deltas = [numeric(arm_rows[k], key) - numeric(base_rows[k], key) for k in keys]
    within = sum(1 for diff in deltas if abs(diff) <= 1)
    return {"pairs": len(deltas), "mean": statistics.fmean(deltas),
            "share_within_one": within / max(1, len(deltas)),
            "share_at_least_two": sum(1 for d in deltas if abs(d) >= 2) / max(1, len(deltas)),
            "max_abs": max(abs(diff) for diff in deltas)}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/v11/r2/p05/ab_three_arm.json"))
    args = parser.parse_args()
    data, tags = {}, {}
    for name in ARMS:
        data[name] = load_arm(name)
        tags[name] = check_tags(name, data[name])
    report = {"arms": {}, "tags": tags}
    for name, rows in data.items():
        per_seed = {}
        for seed in SEEDS:
            subset = {key: entry for key, entry in rows.items() if key[0] == seed}
            per_seed[str(seed)] = {column: describe(subset, column)
                                   for column in ("search_size", "final_size", "steps",
                                                  "ms_per_step", "wall_ms")}
            per_seed[str(seed)]["pave_ms"] = (
                describe(subset, "wall_ms")["mean"] - describe(subset, "search_ms")["mean"])
        report["arms"][name] = {
            "pooled": {column: describe(rows, column)
                       for column in ("search_size", "final_size", "steps",
                                      "ms_per_step", "wall_ms", "search_ms")},
            "pave_ms_pooled": (describe(rows, "wall_ms")["mean"]
                               - describe(rows, "search_ms")["mean"]),
            "per_seed": per_seed,
        }
    report["paired"] = {
        "A_minus_C": {"search_size": paired(data["A"], data["C"], "search_size"),
                      "steps": paired(data["A"], data["C"], "steps")},
        "B_minus_A": {"search_size": paired(data["B"], data["A"], "search_size"),
                      "steps": paired(data["B"], data["A"], "steps")},
        "B_minus_C": {"search_size": paired(data["B"], data["C"], "search_size"),
                      "steps": paired(data["B"], data["C"], "steps")},
    }
    report["drift_vs_C"] = {name: drift(data[name], data["C"])
                            for name in ("A", "B")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print("| 臂 | 前导解码 | PAVE | search | final | steps | ms/步 | PAVE ms |")
    print("|---|---|---|---|---|---|---|---|")
    for name, flags in (("C", "on"), ("A", "off"), ("B", "off")):
        arm = report["arms"][name]
        pave = "on" if name != "B" else "off"
        print("| {} | {} | {} | {:.4f} | {:.4f} | {:.1f} | {:.3f} | {:.1f} |".format(
            name, flags, pave, arm["pooled"]["search_size"]["mean"],
            arm["pooled"]["final_size"]["mean"], arm["pooled"]["steps"]["mean"],
            arm["pooled"]["ms_per_step"]["mean"], arm["pave_ms_pooled"]))
    for label, entry in report["paired"].items():
        stats = entry["search_size"]
        print("{}: Δsearch {:+.4f} [{:+.4f}, {:+.4f}] win/loss/tie {}/{}/{}  Δsteps {:+.1f}".format(
            label, stats["mean"], stats["ci"][0], stats["ci"][1], stats["wins"],
            stats["losses"], stats["ties"], entry["steps"]["mean"]))
    for name, entry in report["drift_vs_C"].items():
        print("{} vs C 图级 search_size: |Δ|<=1 {:.2%}, |Δ|>=2 {:.2%}, max {:.1f}".format(
            name, entry["share_within_one"], entry["share_at_least_two"],
            entry["max_abs"]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
