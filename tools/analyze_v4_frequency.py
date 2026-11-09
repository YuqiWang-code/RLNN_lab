#!/usr/bin/env python3
"""Aggregate completed v4 evaluations without mixing splits or precision protocols."""
import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/v4_frequency"
ARMS = {"none": "RLNN", "v4a_mlp": "状态MLP", "v4a_message": "普通消息", "v4a": "频域适配器"}
METRICS = {
    "mean_size": "Mean Size", "mean_post_size": "冻结PAVE后Mean Size",
    "mean_pave_gain": "冻结PAVE增益", "wall_time_s": "任务墙钟（秒）", "params": "Params",
    "adapter_params": "适配器参数", "peak_memory_mib": "PyTorch峰值已分配显存（MiB）",
    "completed_epochs": "完成轮数", "updates": "更新数", "best_validation_size": "最佳验证Size",
    "maximum_reconstruction_relative": "最大频带相对重构误差", "spectral_graphs": "谱诊断图数",
    "samples": "图数", "graphs": "图数", "states": "状态数", "actions": "动作数",
    "informative_states": "非平局状态数", "informative_graphs": "非平局图数",
    "non_tied_pairs": "非平局候选对", "total_pairs": "候选对总数",
    "informative_state_fraction": "非平局状态比例", "non_tied_pair_fraction": "非平局候选对比例",
    "state_mean_pair_accuracy": "非平局状态平均排序准确率",
    "pooled_pair_accuracy": "非平局候选对排序准确率", "mean_top1_gain": "Top-1平均增益",
    "mean_uniform_gain": "均匀动作平均增益", "mean_oracle_flip_gain": "Oracle翻转平均增益",
    "mean_top1_regret": "Top-1平均后悔值",
}


def family(value):
    value = str(value or "").replace("\\", "/")
    if "/outputs/" in value:
        value = "outputs/" + value.split("/outputs/", 1)[1]
    return re.sub(r"seed\d+", "seed{n}", value)


def average(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return mean(values) if len(values) == len(rows) and values else None


def display_name(meta, identity):
    split = {"validation": "验证", "test": "测试", "train": "训练"}.get(identity["split"], identity["split"])
    precision = "FP32" if identity["precision"] == "no" else identity["precision"].upper()
    protocol = f'{identity["time_budget_s"]:g}s' if identity["time_budget_s"] is not None else "固定量"
    label = {"baseline": "原骨干", "tuned": "微调骨干"}.get(identity["label"], identity["label"])
    if label == identity["arm"]:
        label = ARMS.get(identity["arm"], label)
    if identity["extended_search"]:
        label += "持续搜索"
    post = {"none": "无后处理", "rule23": "规则交换", "pave": "冻结PAVE"}[identity["post"]]
    interventions = [name for name, enabled in identity["interventions"].items() if enabled]
    suffix = " " + ",".join(interventions) if interventions else ""
    return f"{split} {precision} {protocol} {label} + {post}{suffix}"


def read_eval(path, meta):
    config = meta.get("config", {})
    with (path.parent / "per_graph.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != meta["samples"] or len({row["graph_id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: completed graph count does not match per_graph.csv")
    identity = dict(
        label=meta.get("method") or config.get("label") or meta.get("frequency", "none"),
        arm=meta.get("frequency", config.get("frequency", "none")),
        checkpoint_family=family(meta.get("backbone_checkpoint", config.get("save_dir"))),
        cohort=family(meta.get("cohort_id")), split=meta.get("split", config.get("split")),
        precision=config.get("mixed_precision", "unknown"), protocol=meta["protocol"],
        time_budget_s=config.get("time_budget_s"), extended_search=meta.get("extended_search", False),
        post=meta["post"], pave_protocol=meta.get("pave_protocol"), samples=len(rows),
        data_path=family(config.get("test_path")),
        value_checkpoint_family=family(meta.get("value_checkpoint")),
        num_t=config.get("num_t"), num_k=config.get("num_k"),
        horizon=config.get("horizon"), candidate_cap=config.get("candidate_cap"),
        outer_rounds=config.get("outer_rounds"), repair_rounds=config.get("repair_rounds"),
        post_reserve_s=config.get("post_reserve_s"), interventions=meta.get("interventions", {}),
    )
    source = path.parent.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.parent.as_posix()
    values = [int(row["final_size"]) for row in rows]
    run = dict(identity, method=display_name(meta, identity), source=source,
        backbone_seed=meta.get("backbone_seed", meta.get("train_seed")), head_seed=meta.get("head_seed"),
        search_seed=meta.get("search_seed"), mean_size=mean(values),
        mean_initial_size=average(rows, "base_size"), mean_rule_size=average(rows, "rule_size"),
        mean_solve_time_s=average(rows, "solve_time_s"), mean_post_time_s=average(rows, "post_time_s"),
        mean_score_time_s=average(rows, "score_time_s"), mean_search_steps=average(rows, "search_steps"),
        feasible_rate=mean(int(row["conflicts"]) == 0 for row in rows), params=meta.get("params"),
        peak_memory_mib=meta.get("peak_memory_mib"), backbone_checkpoint=meta.get("backbone_checkpoint"),
        value_checkpoint=meta.get("value_checkpoint"), baseline_source=meta.get("baseline_run"),
        delta_base=None, wins=None, ties=None, losses=None,
        max_solve_time_s=max(float(row["solve_time_s"]) for row in rows),
        deadline_overruns=sum(int(row.get("deadline_overrun", 0)) for row in rows),
        post_deadline_hits=sum(int(row.get("post_deadline_hit", 0)) for row in rows),
        incumbent_fallbacks=sum(int(row.get("incumbent_fallback", 0)) for row in rows))
    run["base_gain"] = run["mean_size"] - run["mean_initial_size"]
    run["rule_gain"] = run["mean_size"] - run["mean_rule_size"]
    run["max_deadline_overrun_s"] = max(0, run["max_solve_time_s"] - identity["time_budget_s"]) if identity["time_budget_s"] is not None else None
    run["_identity"] = identity
    run["_rows"] = {row["graph_id"]: row for row in rows}
    run["_invocation"] = path.parent.parent.as_posix()
    return run


def context(run):
    return tuple(run[key] for key in ("split", "precision", "protocol", "time_budget_s", "samples", "data_path", "num_t", "num_k"))


def pair(left, right):
    a, b = left["_rows"], right["_rows"]
    if set(a) != set(b):
        raise ValueError(f'Paired graph IDs differ: {left["source"]}, {right["source"]}')
    shared = left["_invocation"] == right["_invocation"] and left["protocol"] == "fixed_work"
    differences = [int(a[g]["final_size"]) - int(b[g]["final_size"]) for g in a]
    initial = [int(a[g]["base_size"]) - int(b[g]["base_size"]) for g in a]
    if shared and any(initial):
        raise ValueError(f'Shared evaluation starting sizes differ: {left["source"]}')
    return dict(left=left["method"], right=right["method"],
        left_cohort=left["cohort_index"], right_cohort=right["cohort_index"],
        split=left["split"], precision=left["precision"], protocol=left["protocol"],
        seed=left["backbone_seed"], search_seed=left["search_seed"], records=len(a),
        pairing="共享初始解" if shared else "独立搜索，同图同种子", delta_size=mean(differences),
        wins=sum(x > 0 for x in differences), ties=sum(x == 0 for x in differences),
        losses=sum(x < 0 for x in differences), total_gain=sum(differences),
        initial_size_delta=mean(initial), post_gain_difference=mean(differences) - mean(initial),
        left_source=left["source"], right_source=right["source"])


def diagnostic_rows(meta, source):
    kind = {"train": "训练", "diagnostic": "频域诊断", "utility_probe": "动作收益探针"}.get(meta.get("mode"), meta.get("mode", "机制"))
    stage = meta.get("frequency", "")
    for key, title in METRICS.items():
        if key in meta:
            yield [kind, source, stage, title, meta[key]]
    for model, metrics in meta.get("model_metrics", {}).items():
        for key, value in metrics.items():
            yield [kind, source, model, METRICS.get(key, key), value]
    config = meta.get("config", {})
    for key in ("split", "seed", "train_seed", "search_seed"):
        if key in meta:
            yield [kind, source, stage, key, meta[key]]
    for key in ("mixed_precision", "epochs", "batch_size", "graph_batch_size", "lr", "state_source", "freeze_backbone_epochs"):
        if key in config:
            yield [kind, source, stage, key, config[key]]
    if meta.get("mode") == "diagnostic":
        yield [kind, source, stage, "时间口径", "含机制观测开销，不作为正式求解时间"]


def analyze(folder):
    runs, diagnostics = [], []
    for path in sorted(folder.rglob("run.json")):
        if any(part.startswith("smoke") for part in path.relative_to(folder).parts):
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("status") != "completed":
            continue
        if meta.get("mode") == "eval":
            runs.append(read_eval(path, meta))
        elif meta.get("mode") in ("train", "diagnostic", "utility_probe"):
            source = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
            diagnostics.extend(diagnostic_rows(meta, source))
    groups = defaultdict(list)
    for index, run in enumerate(runs):
        run["run_index"] = index
        groups[json.dumps(run["_identity"], sort_keys=True)].append(run)
    cohorts = []
    for entries in groups.values():
        seeds = [run["backbone_seed"] for run in entries]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f'Duplicate backbone seed in configuration: {entries[0]["method"]}')
        for run in entries:
            run["cohort_index"] = len(cohorts)
        row = dict(entries[0]["_identity"], method=entries[0]["method"], seeds=len(entries), seed_ids=sorted(seeds),
                   run_indices=[run["run_index"] for run in entries], params=entries[0]["params"])
        cohorts.append(row)
    pairs = []
    for left, right in combinations(runs, 2):
        if context(left) != context(right) or (left["backbone_seed"], left["search_seed"]) != (right["backbone_seed"], right["search_seed"]):
            continue
        within = left["_invocation"] == right["_invocation"]
        if not within and left["post"] != right["post"] and left["protocol"] != "fixed_time":
            continue
        rank = lambda r: (r["post"] != "none", r["arm"] != "none", r["label"] != "baseline", r["arm"] == "v4a", r["arm"] == "v4a_message", r["method"])
        if rank(left) < rank(right):
            left, right = right, left
        pairs.append(pair(left, right))
    lookup = {run["source"]: run for run in runs}
    for run in runs:
        reference = (run.get("baseline_source") or "").replace("\\", "/").removesuffix("/run.json")
        baseline = lookup.get(reference)
        if baseline is None:
            candidates = [other for other in runs if context(other) == context(run) and other["label"] == "baseline"
                          and other["post"] == run["post"] and (other["backbone_seed"], other["search_seed"]) == (run["backbone_seed"], run["search_seed"])]
            baseline = candidates[0] if len(candidates) == 1 else None
        if baseline is not None and context(baseline) == context(run) and baseline["post"] == run["post"] \
                and (baseline["backbone_seed"], baseline["search_seed"]) == (run["backbone_seed"], run["search_seed"]):
            effect = pair(run, baseline)
            run.update(delta_base=effect["delta_size"], wins=effect["wins"], ties=effect["ties"], losses=effect["losses"], baseline_source=baseline["source"])
    means = ("mean_size", "mean_initial_size", "mean_rule_size", "mean_solve_time_s", "mean_post_time_s", "mean_score_time_s",
             "mean_search_steps", "feasible_rate", "base_gain", "rule_gain", "delta_base")
    sums = ("wins", "ties", "losses", "deadline_overruns", "post_deadline_hits", "incumbent_fallbacks")
    for cohort in cohorts:
        entries = [runs[i] for i in cohort["run_indices"]]
        for key in means:
            cohort[key] = mean(r[key] for r in entries) if all(r[key] is not None for r in entries) else None
        for key in sums:
            cohort[key] = sum(r[key] for r in entries) if all(r[key] is not None for r in entries) else None
        for key in ("peak_memory_mib", "max_solve_time_s", "max_deadline_overrun_s"):
            cohort[key] = max(r[key] for r in entries) if all(r[key] is not None for r in entries) else None
        cohort["std_size"] = stdev(r["mean_size"] for r in entries) if len(entries) > 1 else None
        cohort["by_seed"] = {r["backbone_seed"]: r["mean_size"] for r in entries}
    pair_groups = defaultdict(list)
    for row in pairs:
        pair_groups[row["left_cohort"], row["right_cohort"]].append(row)
    for entries in pair_groups.values():
        row = dict(entries[0], seed="all", search_seed=None, left_source=None, right_source=None)
        for key in ("records", "wins", "ties", "losses", "total_gain"):
            row[key] = sum(r[key] for r in entries)
        for key in ("delta_size", "initial_size_delta", "post_gain_difference"):
            row[key] = mean(r[key] for r in entries)
        row["paired_seeds"] = len(entries)
        pairs.append(row)
    for row in pairs:
        stage = f'{row["left"]} − {row["right"]} / 种子{row["seed"]}'
        for key, title in (("pairing", "配对口径"), ("records", "图—种子记录数"), ("delta_size", "平均Size差"),
                           ("wins", "胜"), ("ties", "平"), ("losses", "负"), ("initial_size_delta", "初始Size差"), ("post_gain_difference", "后处理增益差")):
            diagnostics.append(["逐图配对", row["left_source"] or "已完成种子合并", stage, title, row[key]])
    clean = [{key: value for key, value in run.items() if not key.startswith("_")} for run in runs]
    return dict(version="v4_frequency", updated_at=datetime.now(timezone.utc).isoformat(), runs=clean,
                cohorts=cohorts, pairs=pairs, diagnostics=diagnostics)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze(args.input)
    analysis = args.output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("per_seed.csv", report["runs"]), ("evaluation_summary.csv", report["cohorts"]), ("paired_effects.csv", report["pairs"])):
        write_csv(analysis / name, rows)
    lines = ["RLNN v4 Frequency", "标准差：已完成骨干种子均值的样本标准差；单种子不报告标准差。",
             "协议按划分、精度、预算、骨干来源分开；冻结PAVE与无后处理分别记录。",
             "显存：PyTorch峰值已分配显存，不含CUDA上下文和保留缓存。", ""]
    for c in report["cohorts"]:
        size = f'{c["mean_size"]:.5f}' + (f' ± {c["std_size"]:.5f}' if c["std_size"] is not None else "")
        lines.append(f'{c["method"]}: {size}; seeds={c["seed_ids"]}; n/seed={c["samples"]}; time={c["mean_solve_time_s"]:.5f}s; feasible={c["feasible_rate"]:.5f}')
    if not report["cohorts"]:
        lines.append("尚无已完成正式评测记录。")
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f'V4: {len(report["runs"])} evaluations, {len(report["cohorts"])} configurations, {len(report["pairs"])} paired rows')


if __name__ == "__main__":
    main()
