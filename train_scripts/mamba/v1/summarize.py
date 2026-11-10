"""Summarize v1 MIS results with graph-paired uncertainty and admission gates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3] / "outputs/static_prior_v1"
ARMS = ("baseline", "constants", "centered", "masked")
PROTOCOLS = ("t100", "t1000", "budget05", "budget10")


def load(folder):
    return json.loads((folder / "run.json").read_text(encoding="utf-8"))


def effect(delta, draws):
    delta = np.asarray(delta)
    return dict(delta=float(delta.mean()), ci95=np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist(),
                wins=int((delta > 0).sum()), ties=int((delta == 0).sum()), losses=int((delta < 0).sum()))


def summarize(root, ref, protocols=PROTOCOLS):
    results, pairs, repeat_effects, runs = [], [], [], {}
    graph_ids = None
    for protocol in protocols:
        folder = root / f"{protocol}_ref{ref}"
        data = load(folder)
        runs[protocol] = data
        if data["status"] != "completed":
            raise ValueError(f"{protocol} is incomplete")
        if graph_ids is None:
            graph_ids = data["graph_ids"]
        if graph_ids != data["graph_ids"] or len(graph_ids) != 50:
            raise ValueError("Expected the same 50 validation graphs")
        reference = next(iter(runs.values()))
        for field in ("checkpoint_sha256", "backbone_weight_sha256", "pave_sha256", "manifests", "precision"):
            if data[field] != reference[field]:
                raise ValueError(f"{field} differs for {protocol}")
        if data["checkpoint_sha256"]["centered"] != data["checkpoint_sha256"]["masked"]:
            raise ValueError("The node-term intervention requires the same centered checkpoint")
        with (folder / "per_graph.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        by_key = {(r["arm"], int(r["repeat"]), r["graph_id"]): r for r in rows}
        repeats = data["repeats"]
        expected = {(arm, repeat, graph) for arm in ARMS for repeat in range(repeats) for graph in graph_ids}
        if set(by_key) != expected or len(rows) != len(expected):
            raise ValueError(f"Missing or duplicate graph pairs for {protocol}")
        if any(int(by_key[arm, repeat, graph]["search_seed"]) != 1017 + data["offset"] + i
               for i, graph in enumerate(graph_ids) for arm in ARMS for repeat in range(repeats)):
            raise ValueError(f"Search seeds differ for {protocol}")
        if data["budget_s"] is None and any(int(r["search_steps"]) != data["num_t"] for r in rows):
            raise ValueError(f"Unequal fixed search work for {protocol}")
        for graph in graph_ids:
            order = [int(by_key[arm, 0, graph]["order"]) for arm in ARMS]
            if sorted(order) != list(range(len(ARMS))):
                raise ValueError("Incorrect within-graph arm order")
            if repeats == 2 and any(int(by_key[arm, 1, graph]["order"]) != len(ARMS)-1-pos
                                    for arm, pos in zip(ARMS, order)):
                raise ValueError("Repeat must reverse the within-graph arm order")
        draws = np.random.default_rng(8241).integers(len(graph_ids), size=(10000, len(graph_ids)))
        def metrics(arm, field):
            return np.array([[float(by_key[arm, rep, gid][field]) for gid in graph_ids] for rep in range(repeats)])

        for post in (("none", "pave") if data["budget_s"] is None else ("pave",)):
            size_key = "final_size" if post == "pave" else "base_size"
            time_key = "solve_s" if post == "pave" else "search_s"
            for arm in ARMS:
                size = metrics(arm, size_key)
                times = metrics(arm, time_key)
                results.append(dict(
                    protocol=protocol, post=post, arm=arm, graphs=len(graph_ids), repeats=repeats,
                    mean_size=float(size.mean()), graph_sd=float(size.mean(0).std(ddof=1)),
                    mean_pave_gain=float((size-metrics(arm, "base_size")).mean()),
                    mean_solve_s=float(times.mean()), p95_solve_s=float(np.quantile(times, .95)),
                    mean_search_s=float(metrics(arm, "search_s").mean()),
                    mean_policy_prepare_s=float(metrics(arm, "policy_prepare_s").mean()),
                    mean_pave_s=float(metrics(arm, "pave_s").mean()) if post == "pave" else 0.,
                    feasible_rate=float(metrics(arm, "feasible").mean()),
                    params=data["params"][arm]+(data["pave_params"] if post == "pave" else 0),
                    peak_mib=float(metrics(arm, "peak_memory_mib").max()),
                    solve_peak_mib=float(metrics(arm, "solve_peak_mib").max()),
                    mean_steps=float(metrics(arm, "search_steps").mean()),
                    overruns=int(metrics(arm, "overrun").sum()), fallbacks=int(metrics(arm, "fallback").sum()),
                    repeat_size=size.mean(axis=1).tolist(), repeat_steps=metrics(arm, "search_steps").mean(axis=1).tolist()))
            for arm, control in (("constants", "baseline"), ("centered", "baseline"), ("masked", "baseline"),
                                 ("centered", "constants"), ("centered", "masked")):
                size_delta = metrics(arm, size_key)-metrics(control, size_key)
                steps_delta = metrics(arm, "search_steps")-metrics(control, "search_steps")
                time_delta = 1000*(metrics(arm, time_key)-metrics(control, time_key))
                matched = sum(by_key[arm, rep, gid]["selected_nodes"] == by_key[control, rep, gid]["selected_nodes"]
                              for rep in range(repeats) for gid in graph_ids) if post == "pave" else None
                pairs.append(dict(protocol=protocol, post=post, arm=arm, control=control,
                                  size=effect(size_delta.mean(0), draws), steps=effect(steps_delta.mean(0), draws),
                                  time_ms=effect(time_delta.mean(0), draws), matched_final_sets=matched))
                if repeats > 1:
                    repeat_effects.append(dict(protocol=protocol, arm=arm, control=control,
                                               size=[effect(delta, draws) for delta in size_delta],
                                               steps=steps_delta.mean(1).tolist()))

    training = []
    for arm in ("constants", "centered"):
        folder = root / f"train_{arm}_ref{ref}"
        meta = load(folder)
        with (folder / "training.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))[-1]
        training.append(dict(arm=arm, wall_s=meta["train_wall_s"], updates=meta["updates"],
                             peak_mib=meta["peak_memory_mib"], params=meta["trainable_params"],
                             alpha=float(history["alpha"]) if history["alpha"] else None,
                             c_add=float(history["c_add"]), c_drop=float(history["c_drop"]),
                             delta_prob_mean_abs=float(history["delta_prob_mean_abs"])))
    return dict(graph_ids=graph_ids, ref=ref, summary=results, pairs=pairs, repeat_effects=repeat_effects,
                training=training)


def gates(result):
    def row(protocol, arm, post="pave"):
        return next(r for r in result["summary"] if (r["protocol"], r["arm"], r["post"]) == (protocol, arm, post))

    def delta(protocol, control):
        return next(r["size"]["delta"] for r in result["pairs"] if
                    (r["protocol"], r["post"], r["arm"], r["control"]) == (protocol, "pave", "centered", control))

    ratio = row("t100", "centered", "none")["mean_solve_s"] / row("t100", "baseline", "none")["mean_solve_s"]
    checks = dict(
        speed=ratio <= 1.08, equal_step=delta("t1000", "constants") > 0,
        budget05=all(x > max(y, z) for x, y, z in zip(row("budget05", "centered")["repeat_size"],
                    row("budget05", "baseline")["repeat_size"], row("budget05", "constants")["repeat_size"])),
        budget10=delta("budget10", "baseline") >= -.02,
        feasible=all(row["feasible_rate"] == 1 for row in result["summary"]))
    return dict(status="advance" if all(checks.values()) else "no_go", checks=checks,
                t100_search_ratio=ratio, t1000_delta_vs_constants=delta("t1000", "constants"),
                budget10_delta_vs_baseline=delta("budget10", "baseline"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--ref", type=int, default=0)
    parser.add_argument("--fixed-only", action="store_true")
    args = parser.parse_args()
    result = summarize(args.root, args.ref, PROTOCOLS[:2] if args.fixed_only else PROTOCOLS)
    if not args.fixed_only:
        result["gates"] = gates(result)
    target = args.root / "analysis"
    target.mkdir(exist_ok=True)
    name = f"{'fixed' if args.fixed_only else 'summary'}_ref{args.ref}.json"
    (target / name).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
