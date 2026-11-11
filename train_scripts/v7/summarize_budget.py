"""Summarize the v7 budget (deadline) evaluation against its pre-registered gates.

Reads ``outputs/v7_tccr/budget/budget_ref{r}.csv``, averages the two ABBA repeats
per (graph, seed, budget, arm), forms the paired difference
``recombine - baseline`` per pair and then per graph, and applies B1-B5 from
``docs/temporary/RLNN_v7_TCCR_Prereg_Budget.md``.

    python train_scripts/v7/summarize_budget.py --root outputs/v7_tccr --refs 0
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
GATE = dict(mean_delta=0.05, loss_share=0.10, recombine_ms=25.0)


def load(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def effect(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return dict(count=0, mean=0.0, ci95=[0.0, 0.0], wins=0, ties=0, losses=0)
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, array.size, size=(DRAWS, array.size))
    return dict(count=int(array.size), mean=float(array.mean()),
                ci95=np.quantile(array[draws].mean(axis=1), [.025, .975]).tolist(),
                wins=int((array > 0).sum()), ties=int((array == 0).sum()),
                losses=int((array < 0).sum()))


def arm_means(rows, budget):
    """Mean final size per (graph, seed, arm) over the ABBA repeats."""
    buckets = {}
    for row in rows:
        if float(row["budget"]) != budget:
            continue
        key = (row["graph_id"], row["search_seed"], row["arm"])
        buckets.setdefault(key, []).append(float(row["final_size"]))
    return {key: float(np.mean(values)) for key, values in buckets.items()}


def budget_block(rows, budget):
    means = arm_means(rows, budget)
    graphs = sorted({key[0] for key in means})
    seeds = sorted({key[1] for key in means})
    per_seed = []
    for graph in graphs:
        for seed in seeds:
            base = means.get((graph, seed, "baseline"))
            rec = means.get((graph, seed, "recombine"))
            if base is not None and rec is not None:
                per_seed.append(rec - base)
    per_graph = {}
    for graph in graphs:
        deltas = [means[(graph, seed, "recombine")] - means[(graph, seed, "baseline")]
                  for seed in seeds
                  if (graph, seed, "baseline") in means and (graph, seed, "recombine") in means]
        if deltas:
            per_graph[graph] = float(np.mean(deltas))
    sizes = {arm: mean(value for key, value in means.items() if key[2] == arm) for arm in ("baseline", "recombine")}
    recombine_ms = max(float(row["recombine_time_s"]) for row in rows
                       if float(row["budget"]) == budget and row["arm"] == "recombine")
    sub = [row for row in rows if float(row["budget"]) == budget]
    return dict(budget=budget, pairs=len(per_seed), graphs=len(per_graph),
                mean_baseline=sizes["baseline"], mean_recombine=sizes["recombine"],
                delta=effect(list(per_graph.values())),
                pair_level=effect(per_seed),
                overruns=sum(int(row["deadline_overrun"]) for row in sub),
                recombine_ms_max=1e3 * recombine_ms,
                steps_baseline=mean(float(row["steps"]) for row in sub if row["arm"] == "baseline"),
                steps_recombine=mean(float(row["steps"]) for row in sub if row["arm"] == "recombine"),
                ms_per_step=mean(float(row["ms_per_step"]) for row in sub))


def gates(blocks):
    out = {}
    for block in blocks:
        tag = f"{block['budget']:g}s"
        delta = block["delta"]
        loss_share = block["delta"]["losses"] / max(block["delta"]["count"], 1)
        out[tag] = dict(
            ci_positive=delta["ci95"][0] > 0,
            mean_ok=delta["mean"] >= GATE["mean_delta"],
            loss_ok=loss_share <= GATE["loss_share"],
            overrun_ok=block["overruns"] == 0 and block["recombine_ms_max"] <= GATE["recombine_ms"],
            mean=delta["mean"], ci95=delta["ci95"], loss_share=loss_share,
            overruns=block["overruns"], recombine_ms=block["recombine_ms_max"])
    statuses = [row["ci_positive"] and row["overrun_ok"] for row in out.values()]
    means_ok = all(row["mean_ok"] for row in out.values())
    loss_ok = all(row["loss_ok"] for row in out.values())
    if all(statuses) and means_ok and loss_ok:
        status = "budget_pass"
    elif any(statuses):
        status = "budget_partial"
    else:
        status = "budget_fail"
    return dict(status=status, thresholds=GATE, budgets=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--refs", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    folder = Path(args.root) / "budget"
    analysis = Path(args.root) / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    pooled = []
    for ref in args.refs:
        path = folder / f"budget_ref{ref}.csv"
        if not path.is_file():
            print(f"ref{ref}: no budget csv")
            pooled.append(dict(ref=ref, status="missing"))
            continue
        rows = load(path)
        budgets = sorted({float(row["budget"]) for row in rows})
        blocks = [budget_block(rows, budget) for budget in budgets]
        verdict = gates(blocks)
        summary = dict(version="v7_tccr", mode="budget", ref=ref, budgets=budgets,
                       blocks=blocks, verdict=verdict)
        (analysis / f"summary_budget_ref{ref}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (analysis / f"decision_budget_ref{ref}.json").write_text(
            json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        pooled.append(dict(ref=ref, status=verdict["status"]))
        for block in blocks:
            print(f"ref{ref} {block['budget']:g}s: baseline {block['mean_baseline']:.4f} "
                  f"recombine {block['mean_recombine']:.4f} "
                  f"delta {block['delta']['mean']:+.4f} "
                  f"ci {[round(x, 4) for x in block['delta']['ci95']]} "
                  f"{block['delta']['wins']}/{block['delta']['ties']}/{block['delta']['losses']}")
        print(f"ref{ref}: {verdict['status']}")
    (analysis / "decision_budget.json").write_text(
        json.dumps(dict(version="v7_tccr", mode="budget", refs=args.refs, results=pooled,
                        thresholds=GATE), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
