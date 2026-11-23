"""Graph-level paired statistics for the v11 result cells.

Reads ``outputs/v11/collect/v11_rows_merged.csv`` (written by
``models/merge_v11_shards.py``), collapses repeated rows of the same graph,
and reports for every arm x budget cell:

* mean ``search_size`` with a 5000-rep graph-level bootstrap 95% CI,
* mean ``final_size`` / ``steps`` / ``ms_per_step`` (report-only),
* paired graph-level comparisons against
  - the historical 20.0685 vector (reconstructed from the v11 ledger, see
    ``docs/temporary/RLNN_v11_Implementation.md`` §7.5: the current build is
    graph-identical to the old run except ``graph_000466`` 20.0 -> 19.5),
  - the RLSA 1 s interpolation 20.084 (constant reference, unpaired),
  - the KaMIS 1 s legacy reading 20.106 (constant reference, unpaired; the
    adopted clock-corrected value 20.1020 @ 3.046 s is reported alongside),
  - other arms (0.2 s vs 1.0 s, seed pairs, step-curve endpoints, C4 h=8 vs h=0).

``win`` = graph strictly better (larger independent set) than the reference,
``tie`` = equal, ``loss`` = worse.

Usage::

    python models/v11_paired_stats.py
"""
import argparse
import csv
import json
import random
import statistics as stats
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF_HIST = 20.0685
REF_RLSA_1S = 20.084          # interpolation, not measured at 1 s (v9/v8 SOTA docs)
REF_KAMIS_LEGACY = 20.106     # README "1 s" reading, clock-contaminated
REF_KAMIS_ADOPTED = 20.1020   # 3.046 s wall clock, the reference used from v11 on
FLIP_GRAPH = "graph_000466"   # the single graph the fp16 build moved 20.0 -> 19.5
REPS = 5000
SEED = 20261118

# run directory -> arm label used in the report / workbook
ARMS = {
    "gate0/a0_steps808_n1000": "A0-1s",
    "t7/a0_s1019_steps808_1000": "A0-1s-s1019",
    "t7/a0_s1021_steps808_1000": "A0-1s-s1021",
    "t7/a02_s1017_1000": "P0-0.2s-s1017",
    "t7/a02_s1019_1000": "P0-0.2s-s1019",
    "t7/a02_s1021_1000": "P0-0.2s-s1021",
    "t7/ladder_s1017_1000": "ladder-s1017",
    "t7/curve_k200_s1017_step130": "curve-130",
    "t7/curve_k200_s1017_step200": "curve-200",
    "t7/curve_k200_s1017_step280": "curve-280",
    "t7/curve_k200_s1017_step350": "curve-350",
    "t7/curve_k200_s1017_step450": "curve-450",
    "gate0/a0_clock1s_n1000": "clock-1s(report-only)",
    "gate0/a0_misconfig_fp32_repro": "misconfig-fp32(report-only)",
    "gate0/a0_steps808_repeat": "legacy-repeat(report-only)",
}
# cells whose budget label comes from the file; these runs hold several budgets
REPORT_ONLY = {label for label in ARMS.values() if "report-only" in label}


def load(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def collapse(rows, field):
    """graph_id -> mean(field) over the repeats of one (run, budget) cell."""
    bucket = defaultdict(list)
    for row in rows:
        if row[field] in ("", None):
            continue
        bucket[row["graph_id"]].append(float(row[field]))
    return {graph: stats.mean(values) for graph, values in bucket.items()}


def boot_ci(values, reps=REPS, seed=SEED):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(stats.mean(rng.choices(values, k=n)) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


HIST = {}


def rnd(value, digits=6):
    return None if value is None else round(value, digits)


def wintieloss(diffs):
    win = sum(1 for d in diffs if d > 1e-9)
    loss = sum(1 for d in diffs if d < -1e-9)
    return win, len(diffs) - win - loss, loss


def histogram(diffs):
    """{delta: count} so a reviewer can rebuild win/tie/loss from the merged CSV."""
    counts = defaultdict(int)
    for diff in diffs:
        counts[round(diff, 6)] += 1
    return {str(key): counts[key] for key in sorted(counts)}


def pair(arm, ref, a, b, kind):
    """Paired diffs where both dicts cover the graphs; else constant reference."""
    graphs = sorted(set(a) & set(b))
    diffs = [a[g] - b[g] for g in graphs]
    win, tie, loss = wintieloss(diffs)
    lo, hi = boot_ci(diffs)
    HIST[(arm, ref)] = histogram(diffs)
    return {
        "arm": arm, "reference": ref, "kind": kind, "graphs": len(graphs),
        "delta": rnd(stats.mean(diffs)), "ci_low": rnd(lo), "ci_high": rnd(hi),
        "win": win, "tie": tie, "loss": loss,
    }


def vs_constant(arm, ref, label, values, constant):
    diffs = [value - constant for value in values]
    win, tie, loss = wintieloss(diffs)
    lo, hi = boot_ci(diffs)
    return {
        "arm": arm, "reference": f"{label}={constant}", "kind": "unpaired_constant",
        "graphs": len(values), "delta": rnd(stats.mean(diffs)), "ci_low": rnd(lo), "ci_high": rnd(hi),
        "win": win, "tie": tie, "loss": loss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=ROOT / "outputs/v11/collect/v11_rows_merged.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/v11/collect")
    args = parser.parse_args()

    rows = load(args.merged)
    by_cell = defaultdict(list)
    sources = defaultdict(set)
    for row in rows:
        if row["run"] not in ARMS:
            continue
        by_cell[(ARMS[row["run"]], row["budget"], row["seed"])].append(row)
        sources[(ARMS[row["run"]], row["budget"], row["seed"])].add(row["run"])

    cells = {}
    for (arm, budget, seed), cell_rows in by_cell.items():
        cells[(arm, budget, seed)] = {
            "run": next(iter(sources[(arm, budget, seed)])),
            "rows": len(cell_rows),
            "search": collapse(cell_rows, "search_size"),
            "final": collapse(cell_rows, "final_size"),
            "steps": collapse(cell_rows, "steps"),
            "ms": collapse(cell_rows, "ms_per_step"),
            "dtype": cell_rows[0]["dtype"], "dense": cell_rows[0]["dense"],
            "fixed": cell_rows[0]["fixed_steps"],
        }

    # ---- historical 20.0685 vector, reconstructed from the §7.5 flip ledger ----
    base = cells[("A0-1s", "1.0", "1017")]["search"]
    hist = {g: (19.5 if g == FLIP_GRAPH else v) for g, v in base.items()}
    assert abs(stats.mean(base.values()) - 20.0690) < 1e-9, stats.mean(base.values())
    assert abs(stats.mean(hist.values()) - REF_HIST) < 1e-9, stats.mean(hist.values())
    assert base[FLIP_GRAPH] == 20.0, base[FLIP_GRAPH]

    # ---- pooled three seeds: per-graph mean over seeds (paired on graphs) ----
    def pooled(namer, budget, seeds):
        per_seed = [cells[(namer(s), budget, s)]["search"] for s in seeds]
        return {g: stats.mean(v[g] for v in per_seed) for g in sorted(per_seed[0])}

    a0_pool = pooled(lambda s: "A0-1s" if s == "1017" else f"A0-1s-s{s}", "1.0", ["1017", "1019", "1021"])
    p0_pool = pooled(lambda s: f"P0-0.2s-s{s}", "0.2", ["1017", "1019", "1021"])

    def labeled(arm, budget):
        return f"{arm}@{budget}s" if arm.startswith("ladder") else arm

    summary = []
    for key in sorted(cells):
        arm, budget, seed = key
        cell = cells[key]
        values = list(cell["search"].values())
        lo, hi = boot_ci(values)
        summary.append({
            "arm": labeled(arm, budget), "budget": budget, "seed": seed, "graphs": len(values),
            "rows": cell["rows"], "run": cell["run"],
            "search": rnd(stats.mean(values)), "search_ci_low": rnd(lo), "search_ci_high": rnd(hi),
            "final": rnd(stats.mean(cell["final"].values())),
            "steps": rnd(stats.mean(cell["steps"].values())),
            "ms_per_step": rnd(stats.mean(cell["ms"].values())),
            "fixed_steps": cell["fixed"], "dtype": cell["dtype"], "dense": cell["dense"],
            "report_only": arm in REPORT_ONLY,
        })
    for arm, vector in (("A0-1s-pooled(3seed)", a0_pool), ("P0-0.2s-pooled(3seed)", p0_pool)):
        if vector is None:
            continue
        values = list(vector.values())
        lo, hi = boot_ci(values)
        summary.append({
            "arm": arm, "budget": "1.0" if "1s" in arm else "0.2", "seed": "1017+1019+1021",
            "graphs": len(values), "rows": 3 * len(values), "run": "(pooled)",
            "search": rnd(stats.mean(values)), "search_ci_low": rnd(lo), "search_ci_high": rnd(hi),
            "final": None, "steps": None, "ms_per_step": None,
            "fixed_steps": None, "dtype": "fp16", "dense": "1", "report_only": False,
        })

    comparisons = []
    for key in sorted(cells):
        arm, budget, seed = key
        if arm in REPORT_ONLY:
            continue
        vector = cells[key]["search"]
        name = labeled(arm, budget)
        comparisons.append(pair(name, "hist-20.0685", vector, hist, "paired_vector"))
        comparisons.append(vs_constant(name, None, "rlsa_1s", list(vector.values()), REF_RLSA_1S))
        comparisons.append(vs_constant(name, None, "kamis_legacy_1s", list(vector.values()), REF_KAMIS_LEGACY))
    for name, vector in (("A0-1s-pooled(3seed)", a0_pool), ("P0-0.2s-pooled(3seed)", p0_pool)):
        if vector is None:
            continue
        comparisons.append(pair(name, "hist-20.0685", vector, hist, "paired_vector"))
        comparisons.append(vs_constant(name, None, "rlsa_1s", list(vector.values()), REF_RLSA_1S))
        comparisons.append(vs_constant(name, None, "kamis_legacy_1s", list(vector.values()), REF_KAMIS_LEGACY))

    # arm-vs-arm paired contrasts
    contrasts = [
        ("P0-0.2s-s1017", ("P0-0.2s-s1017", "0.2", "1017"), ("A0-1s", "1.0", "1017")),
        ("A0-1s-s1019", ("A0-1s-s1019", "1.0", "1019"), ("A0-1s", "1.0", "1017")),
        ("A0-1s-s1021", ("A0-1s-s1021", "1.0", "1021"), ("A0-1s", "1.0", "1017")),
        ("curve-130", ("curve-130", "1.0", "1017"), ("curve-450", "1.0", "1017")),
        ("ladder-0.1s", ("ladder-s1017", "0.1", "1017"), ("A0-1s", "1.0", "1017")),
        ("ladder-0.5s", ("ladder-s1017", "0.5", "1017"), ("A0-1s", "1.0", "1017")),
    ]
    for arm, left, right in contrasts:
        a = cells[left]["search"]
        b = cells[right]["search"]
        other = "A0-1s" if "A0-1s" in right[0] else right[0]
        comparisons.append(pair(arm, f"{other} (paired arm)", a, b, "paired_arm"))
    if p0_pool is not None and a0_pool is not None:
        comparisons.append(pair("P0-0.2s-pooled(3seed)", "A0-1s-pooled(3seed) (paired arm)",
                                p0_pool, a0_pool, "paired_arm"))

    # ---- C4 escape probe (200 graphs, best_size per graph) ----
    escape_rows = load(ROOT / "outputs/v11/t7/escape/escape.csv")
    escape = {}
    for row in escape_rows:
        escape.setdefault((row["start"], row["h"]), {})[row["graph_index"]] = float(row["best_size"])
    for (start, h), vector in sorted(escape.items()):
        values = list(vector.values())
        lo, hi = boot_ci(values)
        summary.append({
            "arm": f"C4-escape-{start}-h{h}", "budget": "0.2", "seed": "1017", "graphs": len(values),
            "rows": len(values), "run": "t7/escape", "search": rnd(stats.mean(values)),
            "search_ci_low": rnd(lo), "search_ci_high": rnd(hi), "final": None, "steps": None,
            "ms_per_step": None, "fixed_steps": None, "dtype": "fp16", "dense": "1",
            "report_only": False,
        })
    comparisons.append(pair("C4-escape-deploy-h8", "C4-escape-deploy-h0 (paired arm)",
                            escape[("deploy", "8")], escape[("deploy", "0")], "paired_arm"))
    comparisons.append(pair("C4-escape-greedy-h8", "C4-escape-greedy-h0 (paired arm)",
                            escape[("greedy", "8")], escape[("greedy", "0")], "paired_arm"))

    # ---- reconcile against the evaluator-side cell_means.json ----
    cell_means = {entry["cell"]: entry for entry in
                  json.loads((ROOT / "outputs/v11/t7/cell_means.json").read_text(encoding="utf-8"))}
    mapping = [
        ("gate0/a0_steps808_n1000(reference)", ("A0-1s", "1.0", "1017")),
        ("a02_s1017_1000", ("P0-0.2s-s1017", "0.2", "1017")),
        ("a02_s1019_1000", ("P0-0.2s-s1019", "0.2", "1019")),
        ("a02_s1021_1000", ("P0-0.2s-s1021", "0.2", "1021")),
        ("a0_s1019_steps808_1000", ("A0-1s-s1019", "1.0", "1019")),
        ("a0_s1021_steps808_1000", ("A0-1s-s1021", "1.0", "1021")),
    ]
    reconciliation = {"source": "outputs/v11/t7/cell_means.json", "checks": []}
    for name, key in mapping:
        # cell_means averages every row; ours averages the per-graph repeat mean
        expected = cell_means[name]["search"]
        got = stats.mean(cells[key]["search"].values())
        reconciliation["checks"].append({"cell": name, "cell_means": expected, "recomputed": got,
                                         "abs_diff": abs(expected - got)})
    for name, budget in (("ladder_s1017_1000", "0.1"), ("ladder_s1017_1000", "0.5")):
        for entry in [e for e in cell_means.values() if e["cell"] == "ladder_s1017_1000"]:
            if str(entry["budget"]) != budget:
                continue
            got = stats.mean(cells[("ladder-s1017", budget, "1017")]["search"].values())
            reconciliation["checks"].append({"cell": f"{name}@{budget}s", "cell_means": entry["search"],
                                             "recomputed": got, "abs_diff": abs(entry["search"] - got)})
    reconciliation["max_abs_diff"] = max(check["abs_diff"] for check in reconciliation["checks"])
    reconciliation["passed"] = reconciliation["max_abs_diff"] < 5e-4
    print(f"reconcile vs cell_means.json: max |diff| = {reconciliation['max_abs_diff']:.6f} "
          f"-> {'PASS' if reconciliation['passed'] else 'FAIL'}")

    payload = {
        "generated_by": "models/v11_paired_stats.py",
        "merged_csv": args.merged.relative_to(ROOT).as_posix(),
        "bootstrap": {"reps": REPS, "unit": "graph", "seed": SEED, "ci": "percentile 2.5/97.5"},
        "metric": "search_size (primary); final_size/steps/ms_per_step report-only",
        "references": {
            "hist-20.0685": {
                "value": REF_HIST,
                "construction": "A0-1s per-graph search_size with graph_000466 set to 19.5; "
                                "ledger docs/temporary/RLNN_v11_Implementation.md §7.5 "
                                "(999/1000 graphs identical, one 0.5 flip)",
                "kind": "paired_vector",
            },
            "rlsa_1s": {"value": REF_RLSA_1S, "kind": "unpaired_constant",
                        "note": "interpolation, no measured 1 s point (RLNN_v9_SOTA_Hunt §1)"},
            "kamis_legacy_1s": {"value": REF_KAMIS_LEGACY, "kind": "unpaired_constant",
                                "note": "README 1 s reading; clock-contaminated, superseded"},
            "kamis_adopted": {"value": REF_KAMIS_ADOPTED, "kind": "reference_only",
                              "note": "3.046 s wall clock, adopted from v11 on"},
        },
        "arm_summary": summary,
        "comparisons": comparisons,
        "delta_histograms": {f"{arm}||{ref}": hist for (arm, ref), hist in HIST.items()},
        "reconciliation": reconciliation,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "paired_stats.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                                                encoding="utf-8")

    with (args.out / "arm_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (args.out / "comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    print(f"{'arm':<26}{'bud':>4}{'seed':>11}{'n':>5}{'search':>9}{'95% CI':>19}{'steps':>9}")
    for row in summary:
        if row["report_only"]:
            continue
        print(f"{row['arm']:<26}{row['budget']:>4}{row['seed']:>11}{row['graphs']:>5}"
              f"{row['search']:>9.4f}  [{row['search_ci_low']:>7.4f},{row['search_ci_high']:>7.4f}]"
              f"{(row['steps'] or 0):>9.1f}")
    print("\ncomparisons vs hist-20.0685 / rlsa_1s:")
    for row in comparisons:
        if row["reference"].startswith(("hist", "rlsa")):
            print(f"  {row['arm']:<26}{row['reference']:<22}{row['delta']:>+8.4f}"
                  f"  [{row['ci_low']:>+7.4f},{row['ci_high']:>+7.4f}]  "
                  f"{row['win']:>4}W/{row['tie']:>4}T/{row['loss']:>4}L")


if __name__ == "__main__":
    main()
