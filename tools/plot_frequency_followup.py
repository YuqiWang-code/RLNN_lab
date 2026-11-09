"""Export graph-aggregated mechanism figures as PNG and PDF."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#315a88", "#54a6a6", "#e5a14b", "#bd5d61"]
LABELS = {"baseline": "B", "tuned": "B+FT", "v4a_mlp": "MLP", "v4a_message": "MSG", "v4a": "F4"}


def finish(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_all(analysis):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    folder = analysis / "figures"
    folder.mkdir(exist_ok=True)
    path = analysis / "stage_by_seed.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df[(df.split == "validation") & (df.model == "v4a") & (df.trajectory == "teacher") &
                (df.variant == "full") & (df.step >= 25)]
        stages = ["high_input", "high_stage0_Linear", "high_stage1_SiLU", "high_stage2_Linear",
                  "high_stage3_LayerNorm", "delta_high", "logits"]
        names = ["Input", "Linear 1", "SiLU", "Linear 2", "LayerNorm", "Gate x alpha", "Logits"]
        grouped = df.groupby("signal").mean(numeric_only=True).reindex(stages)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, prefix, title in zip(axes, ("raw", "centered"), ("Raw spectrum", "Stationary modes removed")):
            bottom = np.zeros(len(stages))
            for band, color, label in zip(("low", "mid", "higher", "extreme"), COLORS,
                                           ("[0, 0.5)", "[0.5, 1)", "[1, 1.5)", "[1.5, 2]")):
                values = grouped[f"{prefix}_{band}_fraction"].to_numpy() * 100
                ax.bar(names, values, bottom=bottom, color=color, label=label)
                bottom += values
            ax.tick_params(axis="x", rotation=35)
            ax.set_title(title)
            ax.set_ylim(0, 101)
            ax.set_ylabel("Spectral energy (%)")
        axes[-1].legend(title="Laplacian eigenvalue", loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.suptitle("F4 high branch: validation graphs, teacher states at t = 25..100")
        finish(fig, folder, "01_stage_spectra")
    path = analysis / "same_state_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df[(df.split == "validation") & (df.precision == "fp32")]
        categories = ["selected_conflict", "selected_clear", "addable", "unique_blocker", "multi_blocker", "release_positive"]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
        gate_data = pd.read_csv(analysis / "spectral_behavior_summary.csv")
        gate_data = gate_data[(gate_data.split == "validation") & (gate_data.model == "v4a") &
                              (gate_data.trajectory == "teacher") & (gate_data.variant == "full")]
        group_names = ["selected_conflicted", "selected_clear", "unselected_free", "unselected_one", "unselected_many", "release_positive"]
        for ax, column, title in zip(axes[0], ("gate_low_mean_mean", "gate_high_mean_mean"), ("Low gate", "High gate")):
            matrix = gate_data.pivot(index="group", columns="step", values=column).reindex(group_names)
            gates = ax.imshow(matrix.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(matrix.columns)), matrix.columns)
            ax.set_yticks(range(len(categories)), [x.replace("_", " ") for x in categories])
            ax.set_title(title)
        fig.colorbar(gates, ax=list(axes[0]), label="Independent sigmoid gate", shrink=.85)
        bound = df[df.variant.isin(["no_low", "no_high"])].delta_probability_mean_mean.abs().max() * 100
        bound = max(bound, .01)
        for ax, method, title in zip(axes[1], ("no_low", "no_high"), ("Disable low", "Disable high")):
            matrix = df[df.variant == method].pivot(index="category", columns="step", values="delta_probability_mean_mean").reindex(categories) * 100
            im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
            ax.set_xticks(range(len(matrix.columns)), matrix.columns)
            ax.set_yticks(range(len(categories)), [x.replace("_", " ") for x in categories])
            ax.set_title(title)
            ax.set_xlabel("Teacher trajectory step")
        fig.colorbar(im, ax=list(axes[1]), label="Change in flip probability (percentage points)", shrink=.85)
        finish(fig, folder, "02_node_conditioned_probability")
    path = analysis / "reliability_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df[(df.candidate_group == "all") & df.study.str.contains("reliability")]
        if not df.empty:
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            columns = ["pooled_half_direction_agreement", "half_spearman_mean", "mean_gain_sem_mean"]
            titles = ["Split-half pair direction agreement", "Split-half Spearman", "Mean action gain SEM"]
            for _, rows in df.groupby(["study", "horizon"], dropna=False):
                rows = rows.sort_values("R")
                for ax, col, title in zip(axes, columns, titles):
                    ax.plot(rows.R, rows[col], marker="o", label=f"H={int(rows.horizon.iloc[0])}")
                    ax.set_xticks([2, 4, 8, 16])
                    ax.set_xlabel("Continuation repeats R")
                    ax.set_title(title)
            axes[0].axhline(.5, color="gray", linestyle="--", linewidth=1)
            axes[0].legend()
            fig.tight_layout()
            finish(fig, folder, "03_label_reliability")
    from summarize_frequency_followup import tables
    ranking = tables("p3_utility", "ranking.csv")
    if not ranking.empty:
        selected = ranking[(ranking.split == "train") & (ranking.R == 8) & (ranking.candidate_group == "all") &
                           ranking.study.str.startswith("groups_") & (ranking.trajectory_model == "B")]
        if not selected.empty:
            graph = selected.groupby(["candidate_sampling", "graph_id", "step", "model"]).top1_regret.mean().unstack("model")
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
            maximum = max(graph[["MSG", "F4"]].max().max() * 1.08, .1)
            for ax, sampling in zip(axes, ("uniform", "stratified")):
                points = graph.loc[sampling].reset_index()
                scatter = ax.scatter(points.MSG, points.F4, c=points.step, cmap="viridis", s=30, alpha=.75)
                ax.plot([0, maximum], [0, maximum], "--", color="gray", linewidth=1)
                ax.set_title(f"{sampling.capitalize()} candidates")
                ax.set_xlabel("MSG top-1 regret")
                ax.set_ylabel("F4 top-1 regret")
                ax.set_xlim(-.02, maximum)
                ax.set_ylim(-.02, maximum)
            fig.colorbar(scatter, ax=axes, label="Trajectory step", shrink=.8)
            finish(fig, folder, "04_paired_regret")
    path = analysis / "historical_quality_time.csv"
    if path.exists():
        df = pd.read_csv(path)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        masks = [(df.precision == "fp16") & (df.protocol == "fixed_work") & (df.post == "none"),
                 (df.precision == "fp16") & (df.protocol != "fixed_work") & (df.post == "pave")]
        colors = {"baseline": "#2c7bb6", "tuned": "#e08431", "v4a_mlp": "#39945c",
                  "v4a_message": "#9b5cad", "v4a": "#ba4842"}
        for ax, mask, title in zip(axes, masks, ("Fixed K20 / T100, no postprocess", "0.5 s budget, frozen PAVE")):
            for _, row in df[mask].iterrows():
                label = LABELS[row.label] + (" extended" if row.extended_search else "")
                offset = (6, -14) if row.label == "baseline" and not row.extended_search else (6, 8)
                color = colors[row.label]
                if row.post == "pave":
                    label += f" ({row.mean_search_steps:.0f} steps, PAVE {row.mean_post_time_s * 1000:.0f} ms)"
                    if row.label == "v4a_mlp":
                        offset = (6, -18)
                ax.errorbar(row.mean_solve_time_s * 1000, row.mean_size, yerr=row.std_size, fmt="o", capsize=3, color=color)
                ax.annotate(label, (row.mean_solve_time_s * 1000, row.mean_size),
                            xytext=offset, textcoords="offset points", fontsize=8)
            ax.set_title(title)
            ax.set_xlabel("Mean solve time (ms)")
            ax.set_ylabel("Mean independent-set size")
            ax.margins(x=.3, y=.3)
        fig.tight_layout()
        finish(fig, folder, "05_quality_time")


if __name__ == "__main__":
    plot_all(Path(__file__).resolve().parents[1] / "outputs/v4_frequency/followup/analysis")
