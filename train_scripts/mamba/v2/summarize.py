"""Compute graph-clustered paired intervals, null bands and v2 stage decisions."""

import argparse
import csv
from itertools import combinations
import json
from math import tanh
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "models"))
from static_prior_protocol import AMPLITUDES, ORACLE_DOSES, P0_ARMS


ROOT = REPO / "outputs/mamba_v2"
ORACLES = tuple(f"{name}@{dose}" for name in ("oracle-kamis", "oracle-b") for dose in ORACLE_DOSES)
BUDGET_PROTOCOLS = {"budget05/periodic": .5, "budget10/periodic": 1.}
REQUIRED_PROTOCOLS = {("t100", "endpoint"), ("t400", "endpoint"), ("budget05", "periodic"), ("budget10", "periodic")}


def interval(values, indices):
    samples = values[indices].mean(1)
    lo, hi = np.quantile(samples, [.025, .975])
    return [float(lo), float(hi)]


def compare(left, right, indices):
    values = left - right
    graphs = values.mean(axis=(1, 2))
    return dict(delta=float(graphs.mean()), ci95=interval(graphs, indices),
                se=float(graphs.std(ddof=1) / np.sqrt(len(graphs))) if len(graphs) > 1 else None,
                repeat_delta=values.mean(axis=(0, 2)).tolist(),
                win=int((graphs > 0).sum()), tie=int((graphs == 0).sum()), loss=int((graphs < 0).sum()))


def summarize_group(rows, meta):
    seeds = {key: list(map(int, values)) for key, values in meta["search_seeds"].items()}
    graphs = meta["graph_ids"]
    repeats = range(meta["config"]["repeats"])
    arm_rows = {}
    for row in rows:
        key = row["graph_id"], int(row["repeat"]), int(row["search_seed"])
        indexed = arm_rows.setdefault(row["arm"], {})
        if key in indexed or row["feasible"] != "1":
            raise ValueError(f"Duplicate or infeasible row for {row['arm']} {key}")
        indexed[key] = row
    expected = {(gid, rep, seed) for gid in graphs for rep in repeats for seed in seeds[gid]}
    if "b0" not in arm_rows or any(indexed.keys() != expected for indexed in arm_rows.values()):
        raise ValueError("Arms must share the complete locked graph/seed/repeat matrix")
    keys = [(gid, rep, seed) for gid in graphs for rep in repeats for seed in seeds[gid]]
    shape = len(graphs), len(repeats), len(seeds[graphs[0]])
    indices = np.random.default_rng(1017).integers(len(graphs), size=(5000, len(graphs)))

    def values(arm, field):
        return np.array([float(arm_rows[arm][key][field]) for key in keys]).reshape(shape)

    if "amp@0" in arm_rows and rows[0]["protocol"].startswith("t"):
        for key in keys:
            for field in ("base_nodes", "selected_nodes"):
                if arm_rows["amp@0"][key][field] != arm_rows["b0"][key][field]:
                    raise ValueError(f"Zero-dose identity failed for {key} {field}")
    base_steps = values("b0", "search_steps").mean()
    report = {}
    for field in ("base_size", "final_size"):
        base = values("b0", field)
        null_effects = {}
        null_magnitude = nb = None
        if shape[2] >= 2 and "null-rand" in arm_rows:
            seed_diffs = [base[:, :, i] - base[:, :, j] for i, j in combinations(range(shape[2]), 2)]
            for (i, j), diffs in zip(combinations(range(shape[2]), 2), seed_diffs):
                graphs_delta = diffs.mean(axis=1)
                null_effects[f"seed_{i+1}_vs_{j+1}"] = dict(delta=float(graphs_delta.mean()),
                                                           ci95=interval(graphs_delta, indices))
            random = values("null-rand", field)
            null_effects["null-rand"] = compare(random, base, indices)
            nb = max(abs(effect["delta"]) for effect in null_effects.values())
            null_magnitude = dict(seed_swap=float(np.abs(seed_diffs).mean()),
                                  random_direction=float(np.abs(random-base).mean()))
        arms = {}
        for arm in arm_rows:
            sizes = values(arm, field)
            times = values(arm, "solve_s")
            graph_sizes = sizes.mean(axis=(1, 2))
            arms[arm] = dict(**compare(sizes, base, indices), mean_size=float(graph_sizes.mean()),
                             std_size=float(graph_sizes.std(ddof=1)) if len(graphs) > 1 else None,
                             mean_s=float(times.mean()), p95_s=float(np.quantile(times, .95)),
                             steps=float(values(arm, "search_steps").mean()),
                             step_loss=float(1-values(arm, "search_steps").mean()/base_steps) if base_steps else None,
                             prepare_s=float(values(arm, "policy_prepare_s").mean()),
                             pave_s=float(values(arm, "pave_s").mean()), feasible_rate=1.,
                             params=meta["params"][arm], peak_memory_mib=float(values(arm, "peak_memory_mib").max()),
                             solve_peak_mib=float(values(arm, "solve_peak_mib").max()),
                             bias_rms=float(values(arm, "bias_rms").mean()),
                             node_bias_rms=float(values(arm, "node_bias_rms").mean()),
                             alpha=meta["alpha"].get(arm),
                             delta_prob_mae=float(values(arm, "delta_prob_mae").mean()),
                             boundary_fraction=float(values(arm, "boundary_fraction").mean()),
                             flip_rate=float(values(arm, "flip_rate").mean()),
                             final_jaccard=float(values(arm, "final_jaccard").mean()))
            if rows[0]["protocol"].startswith("budget"):
                arms[arm]["prepare_budget_fraction"] = float(values(arm, "prepare_budget_fraction").mean())
            if all(label in arm_rows for label in ("mamba8", "mlp8", "msg8")) and arm == "mamba8":
                arms[arm]["encoder_pairs"] = {
                    control: compare(sizes, values(control, field), indices) for control in ("mlp8", "msg8")}
            name, _, dose = arm.partition("@")
            control = {"amp": f"rand@{dose}", "oracle-kamis": f"anti-oracle@{dose}"}.get(name)
            if control in arm_rows:
                arms[arm]["direction_pair"] = dict(control=control, **compare(sizes, values(control, field), indices))
        report[field] = dict(null_band=nb, null_effects=null_effects, null_magnitude=null_magnitude, arms=arms)
    return report


def positive(effect, nb):
    return nb is not None and effect["ci95"][0] > nb and all(delta > 0 for delta in effect["repeat_delta"])


def decision(stage, backend, protocols, complete, direction_arm, random_rms=.0107):
    if stage == "p0":
        required = set(P0_ARMS)
        all_protocols = {f"t{step}/{decoder}" for step in (25, 50, 100, 200, 400, 800)
                         for decoder in ("endpoint", "periodic")} | {"budget05/periodic", "budget10/periodic"}
        if not complete or not all_protocols <= protocols.keys() or any(
                block["null_band"] is None or not required <= block["arms"].keys()
                for key in all_protocols for block in protocols[key].values()):
            return dict(status="incomplete")
        curves = [key for key in sorted(all_protocols) if key.startswith("t") and key.endswith("/endpoint")]

        def candidates(keys):
            for protocol in keys:
                for metric, block in protocols[protocol].items():
                    for arm in AMPLITUDES + ORACLES:
                        yield protocol, metric, arm, block["arms"][arm], block["null_band"]

        def passes(effect, nb, arm):
            return positive(effect, nb) and (arm.startswith("oracle-b@")
                                             or positive(effect["direction_pair"], 0.))

        evidence = list(candidates(curves))
        passed = [(protocol, metric, arm) for protocol, metric, arm, effect, nb in evidence
                  if passes(effect, nb, arm)]
        budget_passed = [(protocol, metric, arm)
                         for protocol, metric, arm, effect, nb in candidates(BUDGET_PROTOCOLS)
                         if passes(effect, nb, arm)]
        details = dict(criterion="fixed_step_endpoint", passed=passed, budget_passed=budget_passed)
        oracle_passed = [arm for _, _, arm in passed if arm in ORACLES]
        if oracle_passed:
            name, dose = min((arm.split("@") for arm in oracle_passed),
                             key=lambda item: (float(item[1]), item[0] != "oracle-kamis"))
            details.update(oracle_dose=float(dose), oracle_source=name)
        if doses := [float(arm.split("@")[1]) for _, _, arm in passed if arm in AMPLITUDES]:
            return dict(status="train_p1", node_limit=min(doses), **details)
        if oracle_passed:
            return dict(status="oracle_direction", **details)
        if any(effect["delta"] > nb for _, _, _, effect, nb in evidence):
            return dict(status="inconclusive", **details)
        return dict(status="no_go", **details)
    required = {f"{protocol}/{decoder}" for protocol, decoder in REQUIRED_PROTOCOLS}
    if not complete or not required <= protocols.keys():
        return dict(status="incomplete")
    details = {}
    inert_rms_limit = random_rms / 10
    checks = []
    for protocol in sorted(required):
        block = protocols[protocol]["final_size"]
        arms, nb = block["arms"], block["null_band"]
        if not {"masked", "matched-rand", direction_arm} <= arms.keys() or nb is None:
            return dict(status="incomplete")
        active = arms[direction_arm]
        budget = BUDGET_PROTOCOLS.get(protocol)
        gates = dict(effect=positive(active, nb), resolution=active["se"] is not None and active["se"] <= .02,
                     controls=all(abs(arms[control]["delta"]) <= nb for control in ("masked", "matched-rand")),
                     speed=budget is None or (active["step_loss"] is not None and active["step_loss"] <= .02
                                              and active["p95_s"] <= budget+.005))
        if stage == "p2":
            gates["encoders"] = all(active.get("encoder_pairs", {}).get(name, {}).get("ci95", [0])[0] > 0
                                    for name in ("mlp8", "msg8"))
        checks.append(all(gates.values()))
        alpha = active["alpha"]
        details[protocol] = dict(**gates, alpha=alpha, gate=tanh(alpha) if alpha is not None else None,
                                 bias_rms=active["bias_rms"], node_bias_rms=active["node_bias_rms"])
    passed = all(checks)
    status = ("train_p2" if stage == "p1" else "p2_pass") if passed else "no_go"
    if stage == "p1" and all(not row["effect"] and row["node_bias_rms"] <= inert_rms_limit for row in details.values()):
        status = "inert"
    return dict(status=status, gates=details, inert_node_rms_limit=inert_rms_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="Curve and budget output directories")
    parser.add_argument("--stage", choices=("p0", "p1", "p2"), default="p0")
    parser.add_argument("--ref", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--backend", choices=("fp32-graph", "fp16-eager"), default="fp32-graph",
                        help="Backend of the default run directories")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = args.runs or [ROOT / f"{args.stage}_{kind}_ref{args.ref}_{args.backend}" for kind in ("curve", "budget")]
    results, identity, coverage = {}, {}, {}
    for path in runs:
        meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
        if meta["status"] != "completed" or meta["config"]["ref"] != args.ref:
            raise ValueError(f"Incomplete run or wrong reference seed in {path}")
        backend = meta["backend"]
        locked = {name: meta[name] for name in ("source_sha", "backbone_weight_sha256", "pave_sha256",
                                               "graph_ids", "search_seeds", "checkpoint_sha256", "oracle_sha256",
                                               "limits", "random_rms", "rng_protocol", "alpha", "doses")}
        locked["repeats"] = meta["config"]["repeats"]
        locked["intervention"] = {name: meta["config"][name] for name in ("static_bias_seed", "oracle_amplitude",
                                                                        "feature_drop")}
        if identity.setdefault(backend, locked) != locked:
            raise ValueError(f"Runs for {backend} must share checkpoints, interventions and the graph/seed matrix")
        coverage[backend] = (locked["repeats"] >= (1 if args.stage == "p0" else 2) and len(locked["graph_ids"]) >= 50
                             and all(len(s) >= 3 for s in meta["search_seeds"].values()))
        arms = locked["checkpoint_sha256"]
        active = arms.get("mamba8" if args.stage == "p2" else "centered")
        if args.stage != "p0" and active and any(
                control in arms and arms[control] != active for control in ("masked", "matched-rand")):
            raise ValueError("Mechanism controls must reuse the active checkpoint")
        if args.stage == "p0" and len({value for key, value in arms.items()
                                      if key.startswith(("amp@", "rand@")) or key in ("centered", "v1-node-only")}) > 1:
            raise ValueError("Amplitude interventions must share one checkpoint")
        with (path / "per_graph.csv").open(encoding="utf-8", newline="") as stream:
            groups = {}
            for row in csv.DictReader(stream):
                groups.setdefault(f"{row['protocol']}/{row['decoder']}", []).append(row)
        for protocol, rows in groups.items():
            block = results.setdefault(backend, {})
            if protocol in block:
                raise ValueError(f"Duplicate {backend}/{protocol}")
            block[protocol] = summarize_group(rows, meta)
    report = dict(stage=args.stage, ref=args.ref, backends=results,
                  decisions={backend: decision(args.stage, backend, protocols, coverage[backend],
                                               "mamba8" if args.stage == "p2" else "centered",
                                               identity[backend]["random_rms"])
                             for backend, protocols in results.items()})
    suffix = "" if args.backend == "fp32-graph" else f"_{args.backend}"
    output = args.output or ROOT / f"{args.stage}_summary_ref{args.ref}{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["decisions"], indent=2))


if __name__ == "__main__":
    main()
