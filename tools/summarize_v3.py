#!/usr/bin/env python3
"""Summarize completed v3 runs and their paired per-graph measurements."""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
KINDS = {"opportunities": "动作普查", "labels": "标签采集", "starts": "起始解生成",
         "value_train": "排序训练", "decision": "训练准入"}
VARIANTS = {"D0_fallback": "D0 严格交换后备", "D1_union": "D1 严格交换联合",
            "D2_neutral_h1": "D2 中性动作 H=1", "D3_neutral_h2": "D3 中性动作 H=2",
            "D4_anchored_h2": "D4 固定插入点 H=2", "labels_h1": "部署状态 H=1",
            "labels_h2": "部署状态 H=2", "best": "最佳验证", "validation_rule_gains": "验证规则收益"}
RULE_NAMES = {"coverage": "覆盖度", "degree": "原始度数", "random": "固定随机"}
METRICS = {
    "status": "状态", "split": "数据划分", "samples": "起始解数（图×骨干×随机流）",
    "state_count": "标签状态数", "states": "状态数", "graphs": "不同图数",
    "available": "有候选的状态数", "informative": "非平局状态数", "informative_states": "非平局状态数",
    "informative_graphs": "非平局图数", "non_tied_pairs": "非平局候选对数",
    "positive_states": "Oracle正收益状态数", "positive_coverage": "Oracle正收益覆盖率",
    "mean_candidates": "平均保留候选数", "mean_oracle_gain": "Oracle平均增益",
    "strongest_rule": "最强简单规则", "oracle_gap": "Oracle减最强规则",
    "mean_truncation_loss": "候选截断平均Oracle损失", "mean_cpu_task_s": "平均CPU任务秒数",
    "wall_time_s": "墙钟秒数", "informative_train_states": "训练非平局状态数",
    "informative_train_graphs": "训练非平局图数", "train_states": "训练状态数",
    "val_states": "验证状态数", "train_informative": "训练非平局状态数",
    "train_informative_graphs": "训练非平局图数", "val_informative": "验证非平局状态数",
    "train_seed": "训练种子", "head_seed": "评分种子", "backbone_seed": "骨干种子",
    "search_seed": "搜索种子", "model": "评分模型", "learning_rate": "学习率",
    "epoch": "轮数", "loss": "排序损失", "elapsed_s": "累计秒数", "params": "参数量",
    "selected_gain": "模型所选动作平均增益", "rule_gain": "最强规则平均增益",
    "oracle_gain": "Oracle平均增益", "delta_vs_rule": "模型减最强规则",
    "regret": "Oracle减模型", "positive_oracle_coverage": "Oracle正收益覆盖率",
    "positive_hit_rate": "正收益状态命中率", "informative_pair_accuracy": "非平局对准确率",
    "informative_top1": "非平局状态Top-1", "informative_spearman": "非平局状态Spearman",
    "train": "允许正式训练", "reason": "依据", "passed": "通过准入",
    "validation_graphs": "验证图数", "validation_states": "验证状态数",
    "min_train_states": "训练非平局状态门槛", "min_train_graphs": "训练非平局图数门槛",
    "min_val_graphs": "验证非平局图数门槛", "min_oracle_gap": "Oracle收益差门槛",
}


def readable_value(key, value):
    if key == "strongest_rule":
        return RULE_NAMES.get(value, value)
    if key == "split":
        return {"train": "训练集", "validation": "验证集", "test": "测试集"}.get(value, value)
    if key == "status":
        return {"completed": "已完成", "skipped": "未进入训练", "running": "运行中"}.get(value, value)
    return value


def diagnostic_rows(metadata, source):
    kind = KINDS.get(metadata.get("mode"), metadata.get("mode", "诊断"))
    for key in ("status", "split", "samples", "state_count", "wall_time_s", "mean_cpu_task_s"):
        if key in metadata:
            yield [kind, source, "任务", METRICS.get(key, key), readable_value(key, metadata[key])]
    for variant, measurements in metadata.get("summaries", {}).items():
        scope = VARIANTS.get(variant, variant)
        basis = "原始A.3解，图×骨干×随机流" if variant.startswith(("D0", "D1")) else \
            "规则停止解，图×骨干×随机流" if metadata.get("mode") == "opportunities" else "部署起点与中间状态，按图去重"
        yield [kind, source, scope, "状态口径", basis]
        for key, value in measurements.items():
            if key == "rule_gains":
                for rule, gain in value.items():
                    yield [kind, source, scope, f"{RULE_NAMES.get(rule, rule)}规则平均增益", gain]
                strongest = measurements.get("strongest_rule")
                if strongest in value:
                    yield [kind, source, scope, "最强规则平均增益", value[strongest]]
            else:
                yield [kind, source, scope, METRICS.get(key, key), readable_value(key, value)]
    excluded = {"config", "version", "started_at", "finished_at", "run", "summaries", "mode",
                "status", "split", "samples", "state_count", "wall_time_s", "mean_cpu_task_s"}
    for key, value in scalar_rows({key: value for key, value in metadata.items() if key not in excluded}):
        scope, _, metric = key.rpartition(".")
        scope = "/".join(VARIANTS.get(part, part) for part in scope.split("."))
        label = f"{RULE_NAMES[metric]}规则平均增益" if metric in RULE_NAMES else METRICS.get(metric, metric)
        yield [kind, source, scope or "任务", label, readable_value(metric, value)]


def average(values):
    values = list(values)
    return mean(values) if values and all(value is not None for value in values) else None


def scalar_rows(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from scalar_rows(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from scalar_rows(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def number(row, key):
    value = row.get(key)
    return float(value) if value not in (None, "") else None


def first(mapping, *keys, default=None):
    return next((mapping[key] for key in keys if mapping.get(key) is not None), default)


def read_eval(path, metadata):
    with (path.parent / "per_graph.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    identifiers = [row["graph_id"] for row in rows]
    if len(set(identifiers)) != len(rows) or len(rows) != metadata["samples"]:
        raise ValueError(f"{path}: per_graph.csv must contain one row per reported graph")
    config = metadata.get("config", {})
    budget = first(metadata, "time_budget_s", default=config.get("time_budget_s"))
    protocol = metadata.get("protocol", "time_budget" if budget else "fixed_work")
    method = first(metadata, "label", "method", default=path.parent.name)
    post = metadata.get("post", "none")
    display = metadata.get("display_name", method if post in ("none", None) else f"{method} + {post}")
    sizes = {row["graph_id"]: int(first(row, "final_size", "independent_set_size")) for row in rows}
    run = {
        "source": path.parent.relative_to(ROOT).as_posix(),
        "method": display, "split": metadata.get("split", config.get("split")),
        "protocol": protocol, "backbone_seed": first(metadata, "backbone_seed", "train_seed"),
        "head_seed": metadata.get("head_seed"), "search_seed": metadata.get("search_seed"),
        "samples": len(rows), "mean_size": average(sizes.values()),
        "mean_solve_time_s": average(number(row, "solve_time_s") for row in rows),
        "feasible_rate": average(float(number(row, "conflicts") == 0) for row in rows)
        if all(number(row, "conflicts") is not None for row in rows) else metadata.get("feasible_rate"),
        "params": metadata.get("params"), "trainable_params": metadata.get("trainable_params"),
        "peak_memory_mib": metadata.get("peak_memory_mib"), "time_budget_s": budget,
        "horizon": first(metadata, "horizon", default=config.get("horizon")),
        "candidate_cap": config.get("candidate_cap"), "outer_rounds": config.get("outer_rounds"),
        "baseline_run": metadata.get("baseline_run"), "rule_reference_run": metadata.get("rule_reference_run"),
        "delta_base": None, "delta_rule": None, "wins": None, "ties": None, "losses": None,
        "base_gain": average(sizes[row["graph_id"]] - float(row["base_size"]) for row in rows)
        if all(row.get("base_size") not in (None, "") for row in rows) else None,
        "rule_gain": average(sizes[row["graph_id"]] - float(row["rule_size"]) for row in rows)
        if all(row.get("rule_size") not in (None, "") for row in rows) else None,
    }
    for key in ("search_steps", "neutral_decisions", "score_time_s", "post_time_s", "improving_exchanges"):
        run[key] = average(number(row, key) for row in rows)
    run["deadline_overruns"] = sum(int(float(row["deadline_overrun"])) for row in rows) \
        if all(row.get("deadline_overrun") not in (None, "") for row in rows) else metadata.get("deadline_overruns")
    setup = {key: config.get(key) for key in ("encoder", "neighborhood", "anchored", "horizon",
             "candidate_cap", "outer_rounds", "repair_rounds", "num_t", "num_k", "backbone")}
    run["cohort_id"] = metadata.get("cohort_id", json.dumps(
        [display, run["split"], protocol, budget, run["samples"], setup], sort_keys=True, ensure_ascii=False))
    return run, sizes


def paired(run, reference, sizes, reference_sizes):
    if set(sizes) != set(reference_sizes):
        raise ValueError(f"{run['source']}: paired reference has a different graph set")
    for key in ("split", "backbone_seed", "search_seed"):
        if run.get(key) is not None and reference.get(key) is not None and run[key] != reference[key]:
            raise ValueError(f"{run['source']}: reference differs in {key}")
    differences = [value - reference_sizes[graph] for graph, value in sizes.items()]
    return average(differences), *(sum(test(value) for value in differences)
                                  for test in (lambda v: v > 0, lambda v: v == 0, lambda v: v < 0))


def summarize(folder):
    runs, diagnostics, sizes_by_run = [], [], {}
    for path in sorted(folder.rglob("run.json")):
        if any(part.startswith("smoke_") for part in path.relative_to(folder).parts):
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("status") not in ("completed", "skipped"):
            continue
        source = path.parent.relative_to(ROOT).as_posix()
        if metadata.get("mode") == "eval" and metadata["status"] == "completed":
            run, sizes = read_eval(path, metadata)
            runs.append(run)
            sizes_by_run[source] = sizes
        else:
            diagnostics.extend(diagnostic_rows(metadata, source))

    lookup = {run["source"]: run for run in runs}
    for run in runs:
        for reference_key, delta_key in (("baseline_run", "delta_base"), ("rule_reference_run", "delta_rule")):
            reference_name = run[reference_key]
            if reference_name is None:
                continue
            reference_path = ROOT / reference_name
            if reference_path.name == "run.json":
                reference_path = reference_path.parent
            reference_name = reference_path.relative_to(ROOT).as_posix()
            if reference_name not in lookup:
                source = reference_path / "run.json"
                if not source.exists():
                    continue
                metadata = json.loads(source.read_text(encoding="utf-8"))
                if metadata.get("status") != "completed":
                    continue
                lookup[reference_name], sizes_by_run[reference_name] = read_eval(source, metadata)
            delta, wins, ties, losses = paired(run, lookup[reference_name], sizes_by_run[run["source"]],
                                              sizes_by_run[reference_name])
            run[delta_key] = delta
            if delta_key == "delta_base":
                run.update(wins=wins, ties=ties, losses=losses)

    runs.sort(key=lambda run: (run["cohort_id"], run["backbone_seed"] or 0,
                               run["head_seed"] or 0, run["search_seed"] or 0, run["source"]))
    groups = defaultdict(list)
    for index, run in enumerate(runs):
        groups[run["cohort_id"]].append(index)
    cohorts = []
    for cohort_id, indices in groups.items():
        group = [runs[index] for index in indices]
        scores = [run["mean_size"] for run in group]
        cohort = {key: group[0][key] for key in ("method", "split", "protocol", "time_budget_s",
                  "horizon", "candidate_cap", "outer_rounds", "samples", "params", "trainable_params")}
        cohort.update(cohort_id=cohort_id, run_indices=indices, seeds=len(indices), mean_size=average(scores),
                      std_size=stdev(scores) if len(scores) > 1 else None)
        for key in ("delta_base", "delta_rule", "mean_solve_time_s", "feasible_rate", "base_gain", "rule_gain"):
            cohort[key] = average(run[key] for run in group)
        for key in ("wins", "ties", "losses", "deadline_overruns"):
            values = [run[key] for run in group]
            cohort[key] = sum(values) if all(value is not None for value in values) else None
        peaks = [run["peak_memory_mib"] for run in group]
        cohort["peak_memory_mib"] = max(peaks) if all(value is not None for value in peaks) else None
        cohorts.append(cohort)

    decision_path = folder / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else None
    if decision is not None:
        gate_rows = []
        for key, value in scalar_rows(decision):
            scope, _, metric = key.rpartition(".")
            scope = "/".join(VARIANTS.get(part, part) for part in scope.split("."))
            gate_rows.append(["训练准入", decision_path.relative_to(ROOT).as_posix(), scope or "总体",
                              METRICS.get(metric, metric), readable_value(metric, value)])
        diagnostics = gate_rows + diagnostics
    return dict(version="v3", updated_at=datetime.now(timezone.utc).isoformat(),
                cohorts=cohorts, runs=runs, diagnostics=diagnostics, decision=decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v3")
    args = parser.parse_args()
    report = summarize(args.output.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["RLNN v3 实验汇总", f"时间：{report['updated_at']}",
             "标准差为同配置种子均值的样本标准差；Δ以明确指定的逐图参照计算。", ""]
    for row in report["cohorts"]:
        score = f"{row['mean_size']:.5f}"
        if row["std_size"] is not None:
            score += f" ± {row['std_size']:.5f}"
        lines.append(f"{row['method']} | {row['split']} | {row['protocol']} | n={row['seeds']} | Size={score}"
                     f" | ΔRLNN={row['delta_base']} | Δ规则={row['delta_rule']} | 秒/图={row['mean_solve_time_s']}")
    if not report["cohorts"]:
        lines.append("尚无已完成的求解测试组。")
    if report["decision"] is not None:
        lines.extend(["", "训练准入", json.dumps(report["decision"], ensure_ascii=False, indent=2)])
    lines.extend(["", f"已收录求解任务：{len(report['runs'])}", f"诊断指标：{len(report['diagnostics'])}"])
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"V3 cohorts: {len(report['cohorts'])}, runs: {len(report['runs'])}, diagnostics: {len(report['diagnostics'])}")


if __name__ == "__main__":
    main()
