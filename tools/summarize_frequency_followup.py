"""Aggregate frequency diagnostics by graph and training seed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v4_frequency/followup"


def tables(phase, filename):
    frames = []
    for path in sorted((OUT / phase).rglob(filename)):
        meta_path = path.parent / "run.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "completed" or "smoke" in path.parent.name:
            continue
        frame = pd.read_csv(path)
        config = meta.get("config", {})
        for key in ("split", "seed", "precision", "sampling", "horizon", "trajectory_model"):
            if key not in frame:
                frame[key] = meta.get(key, config.get(key))
        frame["run"] = path.parent.relative_to(OUT).as_posix()
        frame["study"] = re.sub(r"seed[012]", "seed", path.parent.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate(frame, keys, metrics):
    """Within graph, then within seed; Std is over training-seed means."""
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    metrics = [key for key in metrics if key in frame]
    graph = frame.groupby(keys + ["seed", "graph_id"], dropna=False)[metrics].mean().reset_index()
    seeds = graph.groupby(keys + ["seed"], dropna=False)[metrics].mean().reset_index()
    grouped = seeds.groupby(keys, dropna=False)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index(), seeds


def graph_interval(frame, value, seed=20261109):
    """Paired bootstrap resamples graph IDs, keeping all model seeds together."""
    by_graph = frame.groupby("graph_id")[value].mean().to_numpy()
    if not len(by_graph):
        return (None, None)
    rng = np.random.default_rng(seed)
    draws = rng.choice(by_graph, (4000, len(by_graph)), replace=True).mean(1)
    return tuple(np.quantile(draws, [.025, .975]).tolist())


def paired(frame, keys, arm, reference, value):
    pairs, summaries = [], []
    for context, subset in frame.groupby(keys, dropna=False):
        context = context if isinstance(context, tuple) else (context,)
        common = dict(zip(keys, context))
        wide = subset.pivot(index=["seed", "graph_id"], columns=arm, values=value)
        if reference not in wide:
            continue
        for method in wide.columns:
            matched = wide[[method, reference]].dropna() if method != reference else wide[[reference]].dropna()
            delta = (matched[method] - matched[reference]).rename("delta").reset_index()
            delta.loc[delta.delta.abs() < 1e-12, "delta"] = 0.0
            lo, hi = graph_interval(delta, "delta")
            seed_means = delta.groupby("seed").delta.mean()
            seed_means = seed_means.mask(seed_means.abs() < 1e-12, 0.0)
            summaries.append(dict(**common, metric=value, method=method, reference=reference,
                                  mean_delta=delta.delta.mean(), graph_ci_low=lo, graph_ci_high=hi,
                                  wins=int((delta.delta > 0).sum()), ties=int((delta.delta == 0).sum()),
                                  losses=int((delta.delta < 0).sum()), positive_seeds=int((seed_means > 0).sum()),
                                  seed_count=len(seed_means), pairs=len(delta),
                                  by_seed=json.dumps(seed_means.to_dict())))
            for row in delta.to_dict("records"):
                pairs.append(dict(**common, method=method, reference=reference, metric=value, **row))
    return pd.DataFrame(summaries), pd.DataFrame(pairs)


def save(path, frame):
    if not frame.empty:
        frame.to_csv(path, index=False)


def historical():
    report = json.loads((ROOT / "outputs/v4_frequency/summary.json").read_text(encoding="utf-8"))
    rows = [row for row in report["cohorts"] if not row["cohort"] and row["label"] in
            ("baseline", "tuned", "v4a_mlp", "v4a_message", "v4a")]
    return pd.json_normalize(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    analysis = OUT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    stage = tables("p1_spectrum", "stage_spectra.csv")
    if not stage.empty:
        keys = ["split", "model", "trajectory", "variant", "step", "signal"]
        metrics = [c for c in stage if c.endswith(("_fraction", "_energy", "_centroid")) or
                   c in ("total_energy", "energy_per_column", "eigenvalue_max")]
        summary, seeds = aggregate(stage, keys, metrics)
        save(analysis / "stage_summary.csv", summary)
        save(analysis / "stage_by_seed.csv", seeds)
    behavior = tables("p1_spectrum", "gates_and_deltas.csv")
    if not behavior.empty:
        behavior.loc[behavior["group"] == "selected_clear", "corr_B0_x_high_logit_change"] = np.nan
        keys = ["split", "model", "trajectory", "variant", "group", "step"]
        metrics = [c for c in behavior.select_dtypes(include="number").columns
                   if c not in ("seed", "step", "particles", "count")]
        summary, seeds = aggregate(behavior, keys, metrics)
        save(analysis / "spectral_behavior_summary.csv", summary)
        save(analysis / "spectral_behavior_by_seed.csv", seeds)
    replay = tables("p2_branch", "replay_per_graph.csv")
    if not replay.empty:
        repeated = replay[replay.variant.str.contains("_repeat")]
        summary, seeds = aggregate(repeated, ["split", "precision", "variant"], ["base_size", "final_size", "pave_gain"])
        save(analysis / "native_replay_summary.csv", summary)
        replay = replay[~replay.variant.str.contains("_repeat")]
        keys = ["split", "precision", "variant"]
        summary, seeds = aggregate(replay, keys, ["base_size", "final_size", "pave_gain"])
        save(analysis / "replay_summary.csv", summary)
        save(analysis / "replay_by_seed.csv", seeds)
        pairs, detail = [], []
        for metric in ("base_size", "final_size"):
            p, d = paired(replay, ["split", "precision"], "variant", "full", metric)
            pairs.append(p)
            detail.append(d)
        save(analysis / "replay_paired.csv", pd.concat(pairs, ignore_index=True))
        save(analysis / "paired_graph_results.csv", pd.concat(detail, ignore_index=True))
    states = tables("p2_branch", "same_state.csv")
    if not states.empty:
        metrics = ["delta_probability_abs_mean", "delta_probability_mean", "expected_flips", "actual_flips",
                   "top5_jaccard", "delta_logit_rms", "flips_target_error", "probability_below_001",
                   "probability_above_099", "expected_sum_single_flip_delta_energy",
                   "favorable_probability_mass", "logit_immediate_gain_correlation"]
        metrics += ["expected_joint_delta_energy", "sampled_joint_delta_energy", "gate_low_mean", "gate_high_mean",
                    "residual_low_relative", "residual_high_relative", "flips_target_squared_error"]
        repeated = states[states.variant.str.startswith("full_repeat")]
        summary, seeds = aggregate(repeated, ["split", "precision", "variant", "category", "step"], metrics)
        save(analysis / "native_same_state_summary.csv", summary)
        states = states[~states.variant.str.contains("_repeat")]
        summary, seeds = aggregate(states, ["split", "precision", "variant", "category", "step"], metrics)
        save(analysis / "same_state_summary.csv", summary)
        save(analysis / "same_state_by_seed.csv", seeds)
    noise = tables("p2_branch", "native_repeat_noise.csv")
    if not noise.empty:
        for field in ("base_size_delta", "final_size_delta"):
            noise[f"{field}_abs"] = noise[field].abs()
            noise[f"{field}_changed"] = (noise[field] != 0).astype(float)
        for field in ("final_state_changed_entries", "best_archive_changed_entries", "changed_observation_phases"):
            noise[f"{field}_rate"] = (noise[field] > 0).astype(float)
        keys = ["split", "precision", "reference_variant"]
        metrics = [c for c in noise.select_dtypes(include=["number", "bool"]).columns
                   if c not in ("seed", "graph_index", "first_changed_observation_phase")]
        summary, seeds = aggregate(noise, keys, metrics)
        maximum = noise.groupby(keys, dropna=False)[["base_size_delta_abs", "final_size_delta_abs"]].max().add_suffix("_max").reset_index()
        counts = noise.groupby(keys, dropna=False).agg(graph_count=("graph_id", "nunique"), seed_count=("seed", "nunique"), paired_repeat_count=("variant", "size")).reset_index()
        save(analysis / "native_repeat_summary.csv", summary.merge(maximum, on=keys).merge(counts, on=keys))
        save(analysis / "native_repeat_by_seed.csv", seeds)
    save(analysis / "historical_quality_time.csv", historical())
    ranking = tables("p3_utility", "ranking.csv")
    if not ranking.empty:
        keys = ["study", "split", "trajectory_model", "candidate_sampling", "horizon", "R", "candidate_group", "model"]
        metrics = ["pair_accuracy", "top1_gain", "uniform_gain", "oracle_sample_gain", "top1_regret",
                   "top1_positive", "non_tied_pairs", "correct_pair_credit"]
        summary, seeds = aggregate(ranking, keys, metrics)
        pooled = ranking.groupby(keys, dropna=False)[["correct_pair_credit", "non_tied_pairs"]].sum()
        pooled["pooled_pair_accuracy"] = pooled.correct_pair_credit / pooled.non_tied_pairs.replace(0, np.nan)
        counts = ranking.assign(informative=ranking.non_tied_pairs > 0).groupby(keys, dropna=False).agg(
            states=("state_id", "size"), informative_states=("informative", "sum")).reset_index()
        summary = summary.merge(pooled[["pooled_pair_accuracy"]].reset_index(), on=keys).merge(counts, on=keys)
        save(analysis / "utility_summary.csv", summary)
        save(analysis / "utility_by_seed.csv", seeds)
        graph = ranking.groupby(keys + ["seed", "graph_id"], dropna=False).top1_regret.mean().reset_index()
        contrasts, detail = paired(graph, keys[:-1], "model", "MSG", "top1_regret")
        if not contrasts.empty:
            contrasts["regret_improvement_over_MSG"] = -contrasts.mean_delta
            save(analysis / "utility_paired.csv", contrasts)
            save(analysis / "utility_graph_deltas.csv", detail)
    reliability = tables("p3_utility", "reliability.csv")
    if not reliability.empty:
        keys = ["study", "split", "trajectory_model", "candidate_sampling", "horizon", "R", "candidate_group"]
        metrics = ["informative", "zero_replicate_fraction", "mean_gain_sem", "half_spearman", "half_top1_same",
                   "half_top1_set_jaccard", "half_action_sign_agreement", "half_pair_direction_agreement",
                   "half_joint_non_tied_pairs", "non_tied_pairs", "positive_mean_fraction"]
        summary, seeds = aggregate(reliability, keys, metrics)
        pools = reliability.groupby(keys, dropna=False)[["half_direction_match_pairs", "half_joint_non_tied_pairs"]].sum()
        pools["pooled_half_direction_agreement"] = pools.half_direction_match_pairs / pools.half_joint_non_tied_pairs.replace(0, np.nan)
        summary = summary.merge(pools.reset_index(), on=keys)
        save(analysis / "reliability_summary.csv", summary)
        save(analysis / "reliability_by_seed.csv", seeds)
    components = tables("p4_profile", "components.csv")
    if not components.empty:
        graph = components.groupby(["split", "seed", "graph_id", "precision", "model", "component", "unit"], dropna=False).time_s.median().reset_index()
        summary, seeds = aggregate(graph, ["split", "precision", "model", "component", "unit"], ["time_s"])
        save(analysis / "profile_components.csv", summary)
        save(analysis / "profile_components_by_seed.csv", seeds)
    equivalence = tables("p4_profile", "component_equivalence.csv")
    if not equivalence.empty:
        keys = ["split", "precision", "model"]
        metrics = [c for c in equivalence.select_dtypes(include="number").columns
                   if c not in ("seed", "graph_index", "native_forward_repeats")]
        summary, seeds = aggregate(equivalence, keys, metrics)
        maximum = equivalence.groupby(keys, dropna=False)[metrics].max().add_suffix("_max").reset_index()
        save(analysis / "component_equivalence_summary.csv", summary.merge(maximum, on=keys))
    e2e = tables("p4_profile", "end_to_end.csv")
    if not e2e.empty:
        summary, seeds = aggregate(e2e, ["split", "precision", "model", "post"],
                                   ["solve_time_s", "search_time_s", "post_time_s", "final_size", "search_steps"])
        latency = e2e.groupby(["split", "precision", "model", "post"]).solve_time_s.agg(
            median_s="median", p95_s=lambda x: x.quantile(.95), mad_s=lambda x: (x - x.median()).abs().median()).reset_index()
        save(analysis / "profile_e2e.csv", summary.merge(latency, on=["split", "precision", "model", "post"]))
        save(analysis / "profile_e2e_by_seed.csv", seeds)
    runs = []
    for path in sorted(OUT.glob("p*/**/run.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        runs.append(dict(run=path.parent.relative_to(OUT).as_posix(), status=meta.get("status"),
                         wall_time_s=meta.get("wall_time_s"), config=meta.get("config")))
    (analysis / "run_inventory.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"updated_at": datetime.now(timezone.utc).isoformat(), "runs": len(runs),
                      "stage_rows": len(stage), "replay_rows": len(replay), "same_state_rows": len(states)}))
    if args.plots:
        from plot_frequency_followup import plot_all
        plot_all(analysis)


if __name__ == "__main__":
    main()
