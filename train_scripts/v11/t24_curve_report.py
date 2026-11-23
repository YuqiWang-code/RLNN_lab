#!/usr/bin/env python
"""t24: turn the fixed-step runs into the step-quality curve for K=200 / fp16+dense.

Reads the five cells written by ``train_scripts/v11/t24_curve.sh``

    outputs/v11/t7/curve_k200_s1017_step{130,200,280,350,450}/v11_rows.csv

and writes ``outputs/v11/r2/step_quality_k200_fp16.json`` plus the five-line
markdown table used in ``docs/temporary/RLNN_v13_FusedEngine.md``.

Only measurement: no weights, no engine changes, no GPU. Run it with any python
that can read csv (the analysis itself needs nothing else).
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEPS = (130, 200, 280, 350, 450)
CELL = "outputs/v11/t7/curve_k200_s1017_step{}"
OUT = "outputs/v11/r2/step_quality_k200_fp16.json"
RSLA = 19.9620  # RLSA @ 0.201 s, the number this curve is meant to price
A0 = 20.0690  # v11 A0, 810 steps @ 0.867 ms/step


def read(step):
    rows = list(csv.DictReader(open(ROOT / CELL.format(step) / "v11_rows.csv")))
    if not rows:
        raise SystemExit(f"empty artifact for step {step}")
    search = [int(float(r["search_size"])) for r in rows]
    final = [int(float(r["final_size"])) for r in rows]
    steps = [float(r["steps"]) for r in rows]
    wall = [float(r["wall_ms"]) for r in rows]
    per = [float(r["ms_per_step"]) for r in rows]
    search_ms = [float(r["search_ms"]) for r in rows]
    return {
        "step": step,
        "graphs": len(rows),
        "mean_search": round(st.mean(search), 4),
        "mean_final": round(st.mean(final), 4),
        "mean_steps": round(st.mean(steps), 2),
        "mean_wall_ms": round(st.mean(wall), 2),
        "mean_ms_per_step": round(st.mean(per), 4),
        # the honest per-step cost: search clock over the steps actually run
        "search_ms_per_step": round(sum(search_ms) / sum(steps), 4),
        "min_steps": min(steps),
        "max_steps": max(steps),
        "dtype": rows[0].get("dtype"),
        "dense": rows[0].get("dense"),
        "fp32_csr": rows[0].get("fp32_csr"),
        "fixed_steps": rows[0].get("fixed_steps"),
    }


def interpolate(points, x):
    """Piecewise-linear value at x; clamps outside the sampled range."""
    xs = [p["mean_steps"] for p in points]
    ys = [p["mean_search"] for p in points]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    raise AssertionError


def steps_for(points, quality):
    """Smallest step count whose interpolated curve reaches ``quality``."""
    for i in range(len(points) - 1):
        a, b = points[i], points[i + 1]
        if a["mean_search"] <= quality <= b["mean_search"]:
            span = b["mean_search"] - a["mean_search"]
            if span <= 0:
                return a["mean_steps"]
            t = (quality - a["mean_search"]) / span
            return round(a["mean_steps"] + t * (b["mean_steps"] - a["mean_steps"]), 1)
    return None


def fit_log(points):
    """q = a + b*ln(steps): the usual diminishing-returns shape."""
    xs = [math.log(p["mean_steps"]) for p in points]
    ys = [p["mean_search"] for p in points]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    pred = [a + b * x for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - sy / n) ** 2 for y in ys)
    return {"a": round(a, 4), "b": round(b, 4), "r2": round(1 - ss_res / ss_tot, 4)}


def main():
    points = [read(s) for s in STEPS]
    for p in points:
        assert p["dtype"] == "fp16" and p["dense"] == "1" and p["fp32_csr"] == "0", p
        assert abs(p["mean_steps"] - p["step"]) < 0.5, p

    # wall_ms = fixed_ms + marginal * steps: the per-graph cost that does not
    # shrink with the budget (load + prelude + engine bookkeeping). This is what
    # the P0.5 item is allowed to remove; the marginal part is what fusion buys.
    xs = [p["mean_steps"] for p in points]
    ys = [p["mean_wall_ms"] for p in points]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    marginal = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    fixed = (sy - marginal * sx) / n
    per_graph = {
        "fixed_ms": round(fixed, 2),
        "marginal_ms_per_step": round(marginal, 4),
        "share_of_0_2s_search_budget": round(fixed / 160.0, 3),
        "steps_if_fixed_removed": round((160.0 - fixed) / marginal, 1),
    }

    deltas = []
    for i in range(1, len(points)):
        a, b = points[i - 1], points[i]
        deltas.append({
            "from": a["step"],
            "to": b["step"],
            "d_search": round(b["mean_search"] - a["mean_search"], 4),
            "per_10_steps": round(
                (b["mean_search"] - a["mean_search"]) / (b["mean_steps"] - a["mean_steps"]) * 10, 4),
        })

    # the 0.2 s arm measured 130.1 steps at 1.23 ms/step (0.877 forward + 0.35 fixed);
    # a rewrite that reaches 0.55-0.60 ms/step puts 0.16 s at 270-290 steps.
    fast = {}
    for ms in (0.55, 0.575, 0.60):
        n = 0.16 * 1000.0 / ms
        fast[f"{ms:.3f}"] = {"steps": round(n, 1), "quality": round(interpolate(points, n), 4)}

    reach = steps_for(points, RSLA)
    report = {
        "config": {
            "k": 200,
            "dtype": "fp16",
            "dense": True,
            "decode": "end-only (decode_every=1000, top4)",
            "pave": "off (--reserve_frac 0 => final_size == search_size up to 0 rounding)",
            "graphs": 1000,
            "seed": 1017,
            "dataset": "models/data/MIS_tensors/RB_small_author_test",
            "wall_clock": "fixed steps, no deadline",
        },
        "reference": {"rlsa_0_2s": RSLA, "v11_a0_810_steps": A0},
        "points": points,
        "deltas": deltas,
        "fit_log": fit_log(points),
        "per_graph_cost": per_graph,
        "cross_checks": {
            "p0_deadline_130_steps": 19.886,
            "curve_130_steps": points[0]["mean_search"],
            "gap": round(points[0]["mean_search"] - 19.886, 4),
            "v11_a0_810_steps": A0,
            "note": "the 130-step fixed-step rung reproduces the 0.2 s deadline arm (P0)",
        },
        "predicted_270_290": {
            "270": round(interpolate(points, 270), 4),
            "280": round(interpolate(points, 280), 4),
            "290": round(interpolate(points, 290), 4),
        },
        "rewrite_scenarios": fast,
        "steps_needed_for_rlsa": reach,
        "speedup_needed_if_130_steps": (round(reach / 130.1, 2) if reach else None),
        "saturated": bool(deltas[-1]["d_search"] <= 0.01),
        "verdict_rule": {
            "280_ge_19_98": "worth doing (0.2 s par or slightly better than RLSA)",
            "280_approx_19_94": "1.9x is not enough; retreat to 'KaMIS cannot deliver inside 1 s'",
        },
    }
    out = ROOT / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")

    print("| 步数 | mean_search | mean_final | 实际 ms/步（wall） | search ms/步 | 相对上一档 |")
    print("|---|---|---|---|---|---|")
    for i, p in enumerate(points):
        d = "—" if i == 0 else f"{deltas[i - 1]['d_search']:+.4f}"
        print(f"| {p['step']} | {p['mean_search']:.4f} | {p['mean_final']:.4f} | "
              f"{p['mean_ms_per_step']:.3f} | {p['search_ms_per_step']:.3f} | {d} |")
    print()
    print("deltas:", json.dumps(deltas))
    print("fit_log:", json.dumps(report["fit_log"]))
    print("270/280/290:", json.dumps(report["predicted_270_290"]))
    print("rewrite:", json.dumps(fast))
    print("per_graph:", json.dumps(per_graph))
    print("steps for RLSA:", reach, " speedup:", report["speedup_needed_if_130_steps"])
    print("wrote", out)


if __name__ == "__main__":
    main()
