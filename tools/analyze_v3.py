#!/usr/bin/env python3
"""Recompute v3 quality, timing, paired effects and label-origin summaries."""
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/v3"
ANALYSIS = OUTPUT / "analysis"
WITHIN_PAIRS = [("pave", "none"), ("pave", "rule23"), ("random", "rule23"),
                ("pave", "random"), ("pave_untrained", "random"), ("pave", "pave_untrained"),
                ("pave_message", "pave"), ("pave_frequency", "pave"),
                ("pave_frequency", "pave_message"), ("pave_adapted", "pave")]
PAIR_NAMES = {("pave", "random"): "统计PAVE减随机平台",
              ("pave", "pave_untrained"): "训练头减同初始化未训练头",
              ("pave_frequency", "pave_message"): "频率减普通消息",
              ("pave_adapted", "pave"): "适配减原统计头",
              ("pave", "rule23"): "统计PAVE减严格规则",
              ("random", "rule23"): "随机平台减严格规则",
              ("pave_message", "pave"): "普通消息减统计头",
              ("pave_frequency", "pave"): "频率减统计头"}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    if rows:
        with (ANALYSIS / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def load_runs():
    runs = {}
    for directory in sorted(OUTPUT.iterdir()):
        if not directory.is_dir() or not directory.name.startswith(("test_", "budget_")):
            continue
        for path in sorted(directory.glob("*/run.json")):
            meta = json.loads(path.read_text(encoding="utf-8"))
            if meta.get("mode") != "eval" or meta.get("status") != "completed":
                continue
            rows = read_csv(path.parent / "per_graph.csv")
            if len(rows) != meta["samples"] or len({row["graph_id"] for row in rows}) != len(rows):
                raise ValueError(f"{path}: graph count differs from the completed run")
            group = re.sub(r"_seed\d+$", "", directory.name)
            key = group, meta["post"], meta["backbone_seed"]
            runs[key] = dict(meta=meta, rows=rows, source=path.parent.relative_to(ROOT).as_posix())
    return runs


def collect_summary(runs):
    groups = defaultdict(list)
    for (group, policy, seed), run in runs.items():
        groups[group, policy].append((seed, run))
    result = []
    for (group, policy), entries in sorted(groups.items()):
        entries.sort()
        metas = [run["meta"] for _, run in entries]
        rows = [row for _, run in entries for row in run["rows"]]
        sizes = [mean(int(row["final_size"]) for row in run["rows"]) for _, run in entries]
        search_steps = [int(row["search_steps"]) for row in rows]
        solve_times = [float(row["solve_time_s"]) for row in rows]
        budget_slacks = [float(run["meta"]["config"]["time_budget_s"]) - float(row["solve_time_s"])
                         for _, run in entries if run["meta"]["config"].get("time_budget_s") is not None
                         for row in run["rows"]]
        gain = [int(row["final_size"]) - int(row["base_size"]) for row in rows]
        platform = []
        for seed, run in entries:
            rule = runs.get((group, "rule23", seed))
            if rule:
                rule_sizes = {row["graph_id"]: int(row["final_size"]) for row in rule["rows"]}
                platform.extend(int(row["final_size"]) - rule_sizes[row["graph_id"]] for row in run["rows"])
            else:
                platform.extend(int(row["final_size"]) - int(row["rule_size"]) for row in run["rows"])
        by_seed = dict(zip((seed for seed, _ in entries), sizes))
        result.append(dict(group=group, protocol=metas[0]["protocol"], policy=policy,
            graph_seed_records=len(rows), graphs=len({row["graph_id"] for row in rows}), seeds=len(entries),
            mean_size=mean(sizes), std_size=stdev(sizes) if len(sizes)>1 else None,
            seed0=by_seed.get(0), seed1=by_seed.get(1), seed2=by_seed.get(2),
            mean_solve_time_s=mean(float(row["solve_time_s"]) for row in rows),
            mean_post_time_s=mean(float(row["post_time_s"]) for row in rows),
            mean_score_time_s=mean(float(row["score_time_s"]) for row in rows),
            params=metas[0]["params"], head_params=metas[0]["trainable_params"],
            peak_memory_mib=max(meta["peak_memory_mib"] for meta in metas),
            feasible_rate=mean(int(row["conflicts"]) == 0 for row in rows),
            gain_vs_shared_initial=mean(gain), gain_vs_shared_rule=mean(platform),
            improved_vs_initial=sum(value>0 for value in gain), improved_vs_rule=sum(value>0 for value in platform),
            mean_search_steps=mean(search_steps),
            mean_neutral_decisions=mean(int(row["neutral_decisions"]) for row in rows),
            mean_retained_candidates=mean(int(row["candidate_count"]) for row in rows),
            deadline_overruns=sum(int(row["deadline_overrun"]) for row in rows),
            mean_initial_size=mean(int(row["base_size"]) for row in rows),
            mean_rule_size=mean(int(row["rule_size"]) for row in rows),
            min_search_steps=min(search_steps), median_search_steps=median(search_steps),
            max_search_steps=max(search_steps), search_steps_below_100_count=sum(step < 100 for step in search_steps),
            max_solve_time_s=max(solve_times), mean_budget_slack_s=mean(budget_slacks) if budget_slacks else None,
            incumbent_fallbacks=sum(int(row["incumbent_fallback"]) for row in rows),
            post_deadline_hits=sum(int(row["post_deadline_hit"]) for row in rows)))
    return result


def paired_rows(runs, left_group, left, right_group, right):
    seeds = sorted({seed for group, policy, seed in runs if group == left_group and policy == left} &
                   {seed for group, policy, seed in runs if group == right_group and policy == right})
    for seed_label in [*seeds, "all"] if seeds else []:
        a, b = [], []
        for seed in seeds if seed_label == "all" else [seed_label]:
            first = runs[left_group, left, seed]["rows"]
            second = {row["graph_id"]: row for row in runs[right_group, right, seed]["rows"]}
            if {row["graph_id"] for row in first} != set(second):
                raise ValueError(f"Different paired graph sets: {left_group}/{left}, {right_group}/{right}")
            a.extend(first)
            b.extend(second[row["graph_id"]] for row in first)
        shared = left_group == right_group
        if shared and any(x["base_size"] != y["base_size"] for x, y in zip(a, b)):
            raise ValueError(f"Within-run starting sizes differ: {left_group}/{left}, {right}")
        delta = [int(x["final_size"]) - int(y["final_size"]) for x, y in zip(a, b)]
        base_delta = [int(x["base_size"]) - int(y["base_size"]) for x, y in zip(a, b)]
        gains = [(int(x["final_size"]) - int(x["rule_size"]),
                  int(y["final_size"]) - int(y["rule_size"])) for x, y in zip(a, b)]
        grouped = {group: [d for row, d in zip(a, delta) if int(row["group"]) == group] for group in (1, 2)}
        yield dict(left_group=left_group, right_group=right_group, left=left, right=right, seed=seed_label,
            pairing="shared_initial" if shared else "independent_search", graph_seed_records=len(delta),
            delta_size=mean(delta), wins=sum(d>0 for d in delta), ties=sum(d==0 for d in delta),
            losses=sum(d<0 for d in delta), total_gain=sum(delta),
            same_final_nodes=sum(x["selected_nodes"] == y["selected_nodes"] for x, y in zip(a, b)),
            left_only_platform_gain=sum(x>0 and y==0 for x, y in gains),
            right_only_platform_gain=sum(x==0 and y>0 for x, y in gains),
            both_platform_gain=sum(x>0 and y>0 for x, y in gains),
            group1_delta=mean(grouped[1]) if grouped[1] else None,
            group2_delta=mean(grouped[2]) if grouped[2] else None,
            delta_histogram=json.dumps(dict(sorted(Counter(delta).items())), sort_keys=True),
            initial_size_delta=mean(base_delta), same_initial_size=sum(d==0 for d in base_delta),
            post_gain_difference=mean(d-bd for d, bd in zip(delta, base_delta)))


def collect_pairs(runs):
    comparisons = set()
    for group in {key[0] for key in runs}:
        for left, right in WITHIN_PAIRS:
            comparisons.add((group, left, group, right))
    sources = {run["source"]: key for key, run in runs.items()}
    for (group, policy, _), run in runs.items():
        for key in ("baseline_run", "rule_reference_run"):
            reference = run["meta"].get(key)
            if reference and reference.removesuffix("/run.json") in sources:
                reference_group, reference_policy, _ = sources[reference.removesuffix("/run.json")]
                if (group, policy) != (reference_group, reference_policy):
                    comparisons.add((group, policy, reference_group, reference_policy))
    for policy in ("none", "rule23", "random", "pave"):
        comparisons.add(("test_tuned_h1", policy, "test_tuned_h2", policy))
        comparisons.add(("test_tuned_h1", policy, "test_original_h1", policy))
    for left_group, left in (("budget_tuned_message_h1", "pave_message"),
                             ("budget_tuned_frequency_h1", "pave_frequency"),
                             ("budget_tuned_adapted_h1", "pave_adapted")):
        comparisons.add((left_group, left, "budget_tuned_stats_h1", "pave"))
    comparisons.add(("budget_tuned_frequency_h1", "pave_frequency", "budget_tuned_message_h1", "pave_message"))
    for right_group, right in (("budget_tuned_none_ext", "none"), ("budget_tuned_rule23_ext", "rule23")):
        comparisons.add(("budget_tuned_stats_h1", "pave", right_group, right))
    return [row for comparison in sorted(comparisons) for row in paired_rows(runs, *comparison)]


def collect_origins():
    result = []
    for split in ("train", "validation"):
        for horizon in (1, 2):
            path = OUTPUT / f"labels_{split}_h{horizon}" / "per_state.csv"
            if not path.exists():
                continue
            rows = read_csv(path)
            initial_graphs = {row["graph_id"] for row in rows if row["origin"] == "initial" and int(row["informative"])}
            for origin in ("initial", "intermediate", "all"):
                group = rows if origin == "all" else [row for row in rows if row["origin"] == origin]
                informative = [row for row in group if int(row["informative"])]
                graphset = {row["graph_id"] for row in informative}
                gains = {rule: mean(float(row[f"{rule}_gain"]) for row in group)
                         for rule in ("coverage", "degree", "random")}
                strongest = max(gains, key=gains.get)
                oracle = mean(float(row["oracle_gain"]) for row in group)
                result.append(dict(split=split, horizon=horizon, origin=origin, states=len(group),
                    graphs=len({row["graph_id"] for row in group}), informative_states=len(informative),
                    informative_graphs=len(graphset), non_tied_pairs=sum(int(row["non_tied_pairs"]) for row in group),
                    informative_graphs_not_in_initial=len(graphset-initial_graphs),
                    informative_rate=len(informative)/len(group), oracle_gain=oracle, strongest_rule=strongest,
                    strongest_rule_gain=gains[strongest], oracle_gap=oracle-gains[strongest],
                    source=path.relative_to(ROOT).as_posix()))
    return result


def mechanism_data(pairs, origins):
    fixed, budget, sources = {}, {}, {}
    for row in pairs:
        primary = row["left_group"] == row["right_group"] == "test_tuned_h1"
        target = fixed if primary else budget if row["left_group"].startswith("budget_") else None
        if target is None:
            continue
        name = PAIR_NAMES.get((row["left"], row["right"])) if primary else \
            f'{row["left_group"]}/{row["left"]} − {row["right_group"]}/{row["right"]}'
        if not name:
            continue
        seed = "已完成种子合并" if row["seed"] == "all" else f'种子{row["seed"]}'
        target.setdefault(name, {})[seed] = {
            "配对口径": "同任务共享初始解" if primary else "各策略独立搜索，按图与种子配对",
            "图—种子记录数": row["graph_seed_records"], "平均Size差": row["delta_size"],
            "胜": row["wins"], "平": row["ties"], "负": row["losses"], "累计节点差": row["total_gain"],
            "前500图平均差": row["group1_delta"], "后500图平均差": row["group2_delta"],
            "初始Size差": row["initial_size_delta"]}
    for row in origins:
        key = ("训练" if row["split"] == "train" else "验证") + f' H{row["horizon"]}'
        label = {"initial": "起点", "intermediate": "中间状态", "all": "合并"}[row["origin"]]
        sources.setdefault(key, {})[label] = {"状态数": row["states"], "不同图数": row["graphs"],
            "非平局状态数": row["informative_states"], "非平局图数": row["informative_graphs"],
            "非平局候选对数": row["non_tied_pairs"], "新增非平局图数": row["informative_graphs_not_in_initial"],
            "非平局状态比例": row["informative_rate"], "Oracle减最强规则": row["oracle_gap"]}
    return dict(version="v3", status="completed", mode="paired_analysis",
                scope="已完成评测的逐图配对与标签来源分层", updated_at=datetime.now(timezone.utc).isoformat(),
                固定工作量配对=fixed, 固定时间配对=budget, 标签来源分层=sources)


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    summaries, pairs, origins = collect_summary(runs), collect_pairs(runs), collect_origins()
    write_csv("evaluation_summary.csv", summaries)
    write_csv("paired_effects.csv", pairs)
    write_csv("fixed_tuned_h1_summary.csv", [row for row in summaries if row["group"] == "test_tuned_h1"])
    write_csv("fixed_tuned_h1_pairs.csv", [row for row in pairs if row["left_group"] == row["right_group"] == "test_tuned_h1"])
    write_csv("label_origin_summary.csv", origins)
    (ANALYSIS / "run.json").write_text(json.dumps(mechanism_data(pairs, origins), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"V3 analysis: {len(runs)} runs, {len(summaries)} configurations, {len(pairs)} paired rows, {len(origins)} label-origin rows")


if __name__ == "__main__":
    main()
