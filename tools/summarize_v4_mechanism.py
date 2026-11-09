"""Summarize v4 validation mechanisms from the saved diagnostic CSV files."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(root):
    output = []
    directory = root / "outputs/v4_frequency"

    def add(kind, source, scope, metric, value):
        output.append([kind, "outputs/v4_frequency/" + source, scope, metric, value])

    def average(rows, key):
        return mean(float(row[key]) for row in rows)

    def weighted(rows):
        return sum(int(row["count"]) * float(row["mean"]) for row in rows) / sum(int(row["count"]) for row in rows)

    spectral_rows = []
    for seed in range(3):
        name = f"diagnostic_v4a_seed{seed}"
        bands = read_csv(directory / name / "bands.csv")
        source = name + "/bands.csv"
        for key, label in (
            ("alpha_low", "低频残差系数αL"), ("alpha_high", "高频残差系数αH"),
            ("residual_total_relative", "总残差范数/原始隐藏表示范数"),
            ("logit_delta_adapter_rms", "同状态关闭adapter的logit差RMS"),
        ):
            add("残差机制", source, f"F4 seed{seed}；20图×6阶段等权", label, average(bands, key))
        conditioned = read_csv(directory / name / "conditional_distributions.csv")
        source = name + "/conditional_distributions.csv"
        for metric, label in (("gate_low", "gL"), ("gate_high", "gH")):
            for step in (0, 100):
                rows = [row for row in conditioned if row["metric"] == metric and row["group"] == "all" and int(row["step"]) == step]
                add("门控机制", source, f"F4 seed{seed}；t{step}；节点×粒子加权", label + "均值", weighted(rows))
            for group, group_label in (("selected0_blockers0", "未选且c=0"), ("selected0_blockers2plus", "未选且c≥2")):
                rows = [row for row in conditioned if row["metric"] == metric and row["group"] == group and int(row["step"]) >= 25]
                add("门控机制", source, f"F4 seed{seed}；t25–100；{group_label}；节点×粒子加权", label + "均值", weighted(rows))
        rows = [row for row in conditioned if row["metric"] == "probability_delta_adapter" and row["group"] == "all" and int(row["step"]) >= 25]
        probability_change = sum(int(row["count"]) * float(row["mean_abs"]) for row in rows) / sum(int(row["count"]) for row in rows)
        add("残差机制", source, f"F4 seed{seed}；t25–100；节点×粒子加权", "同状态adapter翻转概率平均绝对改变量", probability_change)
        spectral_rows.extend(row for row in read_csv(directory / name / "spectra.csv") if int(row["step"]) >= 25)
    source = "diagnostic_v4a_seed{0,1,2}/spectra.csv"
    for signal in ("L2", "B0", "B1", "high_input", "u_high", "delta_high", "u_low", "delta_low", "logits", "disabled_logits"):
        rows = [row for row in spectral_rows if row["signal"] == signal]
        scope = f"F4；{signal}；t25–100；20图×4阶段×3seed的图级比例等权"
        for key, metric in (("energy_fraction_0_0.5", "λ∈[0,0.5)的谱能量比例"),
                            ("energy_fraction_1_1.5", "λ∈[1,1.5)的谱能量比例"),
                            ("spectral_centroid", "谱重心")):
            add("谱机制", source, scope, metric, average(rows, key))

    state_rows, action_rows, ranking_rows = [], [], []
    for seed in range(3):
        name = f"utility_seed{seed}"
        states = read_csv(directory / name / "states.csv")
        actions = read_csv(directory / name / "actions.csv")
        rankings = read_csv(directory / name / "ranking.csv")
        for row in states + actions + rankings:
            row["seed"] = seed
        state_rows.extend(states)
        action_rows.extend(actions)
        ranking_rows.extend(rankings)
        informative = sum(int(row["informative"]) for row in states)
        pairs = sum(int(row["non_tied_pairs"]) for row in states)
        add("收益探针", name + "/states.csv", f"seed{seed}；{len(states)}状态；两流20步teacher+A.3", "信息状态数", informative)
        add("收益探针", name + "/states.csv", f"seed{seed}；{len(states)}状态；两流20步teacher+A.3", "非平局候选对数", pairs)
        for model in ("baseline", "tuned", "mlp", "message", "frequency"):
            rows = [row for row in rankings if row["model"] == model]
            informative_rows = [row for row in rows if int(row["non_tied_pairs"]) > 0]
            add("收益探针", name + "/ranking.csv", f"{model} seed{seed}；信息状态等权", "非平局对排序正确率", average(informative_rows, "pair_accuracy"))
            add("收益探针", name + "/ranking.csv", f"{model} seed{seed}；全部{len(rows)}状态等权", "top-1平均Size收益", average(rows, "top1_gain"))
    source = "utility_seed{0,1,2}/states.csv"
    informative = sum(int(row["informative"]) for row in state_rows)
    pairs = sum(int(row["non_tied_pairs"]) for row in state_rows)
    all_pairs = sum(int(row["pairs"]) for row in state_rows)
    for label, value in (("状态数", len(state_rows)), ("动作数", len(action_rows)),
                         ("信息状态数", informative), ("信息状态比例", informative / len(state_rows)),
                         ("候选对数", all_pairs), ("非平局候选对数", pairs), ("非平局候选对比例", pairs / all_pairs)):
        add("收益探针", source, "3seed合并；单轨迹状态/冻结teacher续20步/两随机流/A.3", label, value)
    source = "utility_seed{0,1,2}/ranking.csv"
    by_model = {}
    for model in ("baseline", "tuned", "mlp", "message", "frequency"):
        rows = [row for row in ranking_rows if row["model"] == model]
        by_model[model] = {(row["seed"], row["state_id"]): row for row in rows}
        informative_rows = [row for row in rows if int(row["non_tied_pairs"]) > 0]
        add("收益探针", source, f"{model}；合并{len(informative_rows)}信息状态等权", "非平局对排序正确率", average(informative_rows, "pair_accuracy"))
        seed_average = mean(average([row for row in informative_rows if row["seed"] == seed], "pair_accuracy") for seed in range(3))
        add("收益探针", source, f"{model}；先状态平均再三seed等权", "非平局对排序正确率", seed_average)
        pooled = sum(float(row["correct_pair_credit"]) for row in rows) / sum(int(row["non_tied_pairs"]) for row in rows)
        add("收益探针", source, f"{model}；合并{pairs}非平局对等权", "非平局对排序正确率", pooled)
        for key, label in (("top1_gain", "top-1平均Size收益"), ("top1_regret", "top-1平均标签内regret")):
            add("收益探针", source, f"{model}；全部{len(rows)}状态等权", label, average(rows, key))
    rows = list(by_model["frequency"].values())
    for key, label in (("uniform_gain", "均匀候选平均Size收益"), ("oracle_flip_gain", "两流标签内最佳候选平均Size收益")):
        add("收益探针", source, f"全部{len(rows)}状态等权；标签内oracle不是期望收益保证", label, average(rows, key))
    for reference in ("baseline", "message", "mlp"):
        same_node = sum(row["chosen_node"] == by_model[reference][key]["chosen_node"] for key, row in by_model["frequency"].items())
        same_gain = sum(float(row["top1_gain"]) == float(by_model[reference][key]["top1_gain"]) for key, row in by_model["frequency"].items())
        add("收益探针", source, f"frequency vs {reference}；{len(rows)}匹配状态", "top-1节点相同状态数", same_node)
        add("收益探针", source, f"frequency vs {reference}；{len(rows)}匹配状态", "top-1实测收益相同状态数", same_gain)
    actions_by_state = {}
    for row in action_rows:
        actions_by_state.setdefault((row["seed"], row["state_id"]), []).append(row)
    both_non_tied = agreement = 0
    for rows in actions_by_state.values():
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                first = float(left["gain_r0"]) - float(right["gain_r0"])
                second = float(left["gain_r1"]) - float(right["gain_r1"])
                if first and second:
                    both_non_tied += 1
                    agreement += first * second > 0
    source = "utility_seed{0,1,2}/actions.csv"
    for label, value in (("两次续跑均非平局的候选对数", both_non_tied),
                         ("两次均非平局且排序同向的对数", agreement),
                         ("两次均非平局条件下的排序同向比例", agreement / both_non_tied)):
        add("收益标签稳定性", source, f"{len(state_rows)}状态；两条共享随机流分别计算候选收益差", label, value)
    return {"diagnostics": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.root / "outputs/v4_frequency/analysis/mechanism_summary.json"
    summary = summarize(args.root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{destination}: {len(summary['diagnostics'])} diagnostic summaries")


if __name__ == "__main__":
    main()
