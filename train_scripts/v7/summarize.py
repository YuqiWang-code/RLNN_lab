"""Summarize the v7 TCCR census and apply the pre-registered gates.

Reads the per-graph CSVs written by ``models/diagnose_v7_census.py`` and applies
the thresholds fixed in ``docs/temporary/RLNN_v7_TCCR_Prereg.md``:

    python train_scripts/v7/summarize.py --root outputs/v7_tccr --refs 0 1 2

Graph-level values are averaged over search seeds first, so a graph never counts
twice.  Intervals are graph-level bootstrap with a fixed RNG seed.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "outputs/v7_tccr"
DRAWS = 2000
RNG_SEED = 20261111
GATE = dict(gain=0.10, graphs=20, pair_us=2000.0, arm_fraction=0.5,
            census_ms=2000.0, census_p95=4000.0, joint_graphs=10)


def load(folder, pattern):
    rows = []
    for path in sorted(Path(folder).glob(pattern)):
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def draws(count):
    return np.random.default_rng(RNG_SEED).integers(0, count, size=(DRAWS, count))


def effect(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return dict(count=0, mean=0.0, ci95=[0.0, 0.0], wins=0, ties=0, losses=0)
    sample = draws(array.size)
    return dict(count=int(array.size), mean=float(array.mean()),
                ci95=np.quantile(array[sample].mean(axis=1), [.025, .975]).tolist(),
                wins=int((array > 0).sum()), ties=int((array == 0).sum()),
                losses=int((array < 0).sum()))


def per_graph(rows, key):
    buckets = {}
    for row in rows:
        buckets.setdefault(row["graph_id"], []).append(float(row[key]))
    return {graph: float(np.mean(values)) for graph, values in buckets.items()}


def per_graph_best(rows, sizes):
    buckets = {}
    for row in rows:
        value = max(float(row[f"gain_{size}"]) for size in sizes)
        buckets.setdefault(row["graph_id"], []).append(value)
    return {graph: float(np.mean(values)) for graph, values in buckets.items()}


def percentile(values, share):
    array = np.sort(np.asarray(values, dtype=float))
    if array.size == 0:
        return 0.0
    index = min(array.size - 1, int(round(share * (array.size - 1))))
    return float(array[index])


def recombine_block(folder, ref, sizes):
    rows = load(folder, f"recombine_ref{ref}*.csv")
    if not rows:
        return None
    gains = per_graph(rows, "oracle_gain")
    oracle = effect(list(gains.values()))
    distinct = per_graph(rows, "distinct")
    union = per_graph(rows, "union_size")
    best = per_graph(rows, "best_size")
    jaccard = per_graph(rows, "jaccard_mean")
    arms = {name: effect(list(per_graph(rows, f"arm_{name}").values()))
            for name in ("random", "hamming", "top_size", "matching_cost")}
    cost_us = mean(float(row["oracle_us_per_pair"]) for row in rows)
    follow = arms["hamming"]["mean"]
    return dict(graphs=len(gains), oracle=oracle,
                improved_graphs=sum(value > 0 for value in gains.values()),
                mean_distinct=mean(distinct.values()), mean_union=mean(union.values()),
                mean_best_size=mean(best.values()), mean_jaccard=mean(jaccard.values()),
                arms=arms, us_per_pair=cost_us,
                hamming_fraction=(follow / oracle["mean"]) if oracle["mean"] else None,
                improving_pairs=mean(float(row["improving_pairs"]) for row in rows),
                useful_pairs=mean(float(row.get("useful_pairs", 0)) for row in rows),
                pairwise_gain_max=mean(float(row.get("pairwise_gain_max", 0)) for row in rows),
                pair_count=mean(float(row["pair_count"]) for row in rows),
                full_oracle_ms=mean(float(row["pair_count"]) for row in rows) * cost_us / 1000)


def conflict_block(folder, ref, sizes, layer, subset=None):
    rows = [row for row in load(folder, f"conflict_ref{ref}*.csv") if row["layer"] == layer]
    if subset is not None:
        rows = [row for row in rows if int(row["stalled"]) == subset]
    if not rows:
        return None
    best = effect(list(per_graph_best(rows, sizes).values()))
    per_size = {}
    for size in sizes:
        gains = per_graph(rows, f"gain_{size}")
        per_size[size] = dict(gain=effect(list(gains.values())),
                              improved_graphs=sum(value > 0 for value in gains.values()),
                              regions=mean(float(row[f"regions_{size}"]) for row in rows),
                              improving=mean(float(row[f"improving_{size}"]) for row in rows))
    census = [float(row["census_ms"]) for row in rows]
    return dict(graphs=len(per_graph(rows, "final_size")), best=best, per_size=per_size,
                census_ms=mean(census), census_p95=percentile(census, .95),
                mean_final_size=mean(float(row["final_size"]) for row in rows),
                mean_pave_gain=mean(float(row["pave_gain"]) for row in rows))


def pipeline_block(root, ref):
    rows = load(Path(root) / "pipeline", f"pipeline_ref{ref}*.csv")
    if not rows:
        return None
    a3 = per_graph(rows, "a3_size")
    merged = per_graph(rows, "recombine_size")
    pave = per_graph(rows, "pave_size")
    both = per_graph(rows, "recombine_pave_size")
    pairs = {graph: merged[graph] - a3[graph] for graph in a3}
    marginal = {graph: both[graph] - pave[graph] for graph in a3}
    pave_gain = {graph: pave[graph] - a3[graph] for graph in a3}
    return dict(graphs=len(a3),
                mean_a3=mean(a3.values()), mean_recombine=mean(merged.values()),
                mean_pave=mean(pave.values()), mean_recombine_pave=mean(both.values()),
                recombine_effect=effect(list(pairs.values())),
                pave_effect=effect(list(pave_gain.values())),
                marginal_over_pave=effect(list(marginal.values())),
                recombine_ms=mean(float(row["recombine_ms"]) for row in rows),
                pave_ms=mean(float(row["pave_ms"]) for row in rows),
                pairs=mean(float(row["pairs"]) for row in rows))


def decision(recombine, conflict, joint_graphs):
    gates = {}
    if recombine is not None:
        gates["G1"] = dict(passed=recombine["oracle"]["mean"] >= GATE["gain"]
                           and recombine["oracle"]["ci95"][0] > 0
                           and recombine["improved_graphs"] >= GATE["graphs"],
                           mean_gain=recombine["oracle"]["mean"],
                           ci95=recombine["oracle"]["ci95"],
                           improved_graphs=recombine["improved_graphs"])
        gates["G2"] = dict(passed=recombine["us_per_pair"] <= GATE["pair_us"]
                           and (recombine["hamming_fraction"] or 0) >= GATE["arm_fraction"],
                           us_per_pair=recombine["us_per_pair"],
                           hamming_fraction=recombine["hamming_fraction"])
    if conflict is not None:
        gates["G3"] = dict(passed=conflict["best"]["mean"] >= GATE["gain"]
                           and conflict["best"]["ci95"][0] > 0
                           and conflict["best"]["wins"] >= GATE["graphs"],
                           mean_gain=conflict["best"]["mean"],
                           ci95=conflict["best"]["ci95"],
                           improved_graphs=conflict["best"]["wins"])
        gates["G4"] = dict(passed=conflict["census_ms"] <= GATE["census_ms"]
                           and conflict["census_p95"] <= GATE["census_p95"],
                           census_ms=conflict["census_ms"], census_p95=conflict["census_p95"])
        gates["G5"] = dict(passed=joint_graphs >= GATE["joint_graphs"],
                           graphs_with_three_to_four=joint_graphs)
    recombine_ok = bool(gates.get("G1", {}).get("passed")) and bool(gates.get("G2", {}).get("passed"))
    conflict_ok = (bool(gates.get("G3", {}).get("passed")) and bool(gates.get("G4", {}).get("passed"))
                   and bool(gates.get("G5", {}).get("passed")))
    if recombine_ok and conflict_ok:
        status = "advance"
    elif recombine_ok:
        status = "recombine_only"
    elif conflict_ok:
        status = "conflict_only"
    else:
        status = "stop"
    return dict(status=status, gates=gates, thresholds=GATE,
                recombine_passed=bool(recombine_ok), conflict_passed=bool(conflict_ok))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--refs", type=int, nargs="+", default=[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()
    folder = args.root / "census"
    analysis = args.root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    pooled = []
    for ref in args.refs:
        recombine = recombine_block(folder, ref, args.sizes)
        conflict = conflict_block(folder, ref, args.sizes, "pave")
        stalled = conflict_block(folder, ref, args.sizes, "pave", subset=1)
        a3 = conflict_block(folder, ref, args.sizes, "a3")
        joint = 0
        if conflict is not None:
            rows = [row for row in load(folder, f"conflict_ref{ref}*.csv") if row["layer"] == "pave"]
            joint = len({row["graph_id"] for row in rows if float(row["gain_3"]) >= 1})
        verdict = decision(recombine, conflict, joint) if recombine or conflict else None
        summary = dict(version="v7_tccr", ref=ref, sizes=args.sizes,
                       recombine=recombine, conflict_pave=conflict, conflict_a3=a3,
                       conflict_stalled=stalled, verdict=verdict)
        summary["pipeline"] = pipeline_block(args.root, ref)
        (analysis / f"summary_ref{ref}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if verdict is not None:
            (analysis / f"decision_ref{ref}.json").write_text(
                json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pooled.append(dict(ref=ref, status=verdict["status"] if verdict else "missing"))
        print(f"ref{ref}: " + (verdict["status"] if verdict else "no data"))
    (analysis / "decision.json").write_text(
        json.dumps(dict(version="v7_tccr", mode="census", refs=args.refs, results=pooled,
                        thresholds=GATE), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
