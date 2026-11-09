"""Summarize v5 replay, residual spectra and utility probes by graph and seed."""
import json
from pathlib import Path

import pandas as pd

from summarize_frequency_followup import aggregate, paired


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v5_frequency"


def tables(name):
    paths = list((OUT / "diagnosis_p1").glob(f"*/{name}.csv"))
    paths += list(OUT.glob(f"diagnostics_*/{name}.csv"))
    frames = []
    for path in sorted(paths):
        meta = json.loads(path.with_name("run.json").read_text(encoding="utf-8"))
        frame = pd.read_csv(path)
        frame["split"] = meta["split"]
        frame["precision"] = meta["precision"]
        frame["initial"] = meta["initial"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(name, keys, metrics, *, late=False):
    frame = tables(name)
    if frame.empty:
        return
    if late:
        frame = frame[frame.step >= 25]
    groups = ["initial", "split", "precision"] + keys
    summary, seeds = aggregate(frame, groups, metrics)
    summary.to_csv(OUT / "analysis" / f"diagnostic_{name}.csv", index=False)
    seeds.to_csv(OUT / "analysis" / f"diagnostic_{name}_by_seed.csv", index=False)


def main():
    summarize("replay", ["arm"], ["base_size", "final_size", "pave_gain", "best_conflicts_mean", "feasible"])
    replay = tables("replay")
    effects = []
    if not replay.empty:
        for control in ("baseline", "same_backbone", "v5a_message", "v5a_mlp"):
            for metric in ("base_size", "final_size"):
                summary, _ = paired(replay, ["initial", "split", "precision"], "arm", control, metric)
                if not summary.empty:
                    effects.append(summary[summary.method == "v5a"])
    if effects:
        pd.concat(effects, ignore_index=True).to_csv(OUT / "analysis" / "diagnostic_replay_effects.csv", index=False)
    summarize("stage_spectra", ["arm", "trajectory", "signal"],
              ["stationary_fraction", "raw_higher_fraction", "centered_higher_fraction", "total_energy",
               "centered_spectral_centroid"], late=True)
    summarize("node_groups", ["variant", "category"], ["delta_probability_abs_mean", "delta_logit_rms",
              "top5_jaccard", "sampled_action_disagreement", "expected_flips", "expected_joint_delta_energy",
              "sampled_joint_delta_energy"], late=True)
    summarize("utility_ranking", ["arm", "sampling"], ["pair_accuracy", "top1_gain", "top1_regret"])
    summarize("utility_reliability", ["sampling"], ["half_spearman", "half_pair_direction_agreement",
              "zero_replicate_fraction", "mean_gain_sem"])
    profile = OUT / "profile_seed0"
    if (profile / "run.json").exists():
        timing = pd.read_csv(profile / "end_to_end.csv")
        timing.groupby("arm").agg(graphs=("graph_id", "nunique"), trials=("repeat", "size"),
            search_time_s=("search_time_s", "mean"), post_time_s=("post_time_s", "mean"),
            total_time_s=("total_time_s", "mean"), total_p95_s=("total_time_s", lambda s: s.quantile(.95))
        ).to_csv(OUT / "analysis" / "profile_end_to_end.csv")
        micro = pd.read_csv(profile / "runtime_micro.csv")
        micro.groupby(["arm", "component", "steps"]).median_s.mean().rename("mean_of_graph_median_s").to_csv(
            OUT / "analysis" / "profile_components.csv")


if __name__ == "__main__":
    main()
