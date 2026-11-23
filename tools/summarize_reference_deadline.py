"""Validate and summarize matched 200 ms MIS baseline results."""
import argparse
from itertools import combinations
import json
from pathlib import Path
from statistics import fmean

from compare_mis_runs import compare, read_rows, summarize


METHODS = ("isco_full_window", "rlsa", "kamis_online", "kamis", "isco")
SEEDS = (1017, 1019, 1021)


def load(root, method, expected, node_counts):
    directory = root / method
    rows = read_rows(directory / "per_graph.csv")
    if rows.keys() != expected:
        raise ValueError(f"{method}: graph/seed rows do not match the expected {len(expected)} runs")

    seen = set()
    with (directory / "solutions.jsonl").open(encoding="utf-8") as records:
        for line in records:
            record = json.loads(line)
            key = record["graph_id"], str(record["seed"]), "0"
            score, wall, ready, _ = rows[key]
            updates = record["updates"]
            last = updates[-1] if updates else None
            if (last is None and (score != 0 or ready is not None)) or (
                last is not None and (len(last["nodes"]) != score or last["ready_ms"] != ready)
            ):
                raise ValueError(f"{method}: solution and scalar record differ for {key}")
            prior_size = -1
            prior_time = 0
            for update in updates:
                size = len(update["nodes"])
                time = update["ready_ms"]
                if not (prior_size < size == len(set(update["nodes"])) and prior_time <= time <= min(wall, 200)
                        and all(0 <= node < node_counts[key[0]] for node in update["nodes"])):
                    raise ValueError(f"{method}: invalid score update for {key}")
                prior_size, prior_time = size, time
            if key in seen:
                raise ValueError(f"{method}: duplicate solution record for {key}")
            seen.add(key)
    if seen != expected:
        raise ValueError(f"{method}: mismatched solution records")

    metrics = summarize(rows, 200)
    metrics["wall_overruns"] = sum(wall > 200 for _, wall, _, _ in rows.values())
    delivered = [size for size, _, ready, _ in rows.values() if ready is not None]
    metrics["mean_size_on_delivery"] = fmean(delivered) if delivered else None
    metrics["per_seed"] = {str(seed): summarize({key: row for key, row in rows.items()
                                               if key[1] == str(seed)}, 200) for seed in SEEDS}
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--follow-up", type=Path, help="directory with author_rlnn and fld_eg runs")
    parser.add_argument("--learned", type=Path, help="directory with official DiffUCO and LTFT runs")
    parser.add_argument("--supervised", type=Path, help="directory with official DIFUSCO and DGL-TreeSearch runs")
    parser.add_argument("--intel-ppo", type=Path, help="directory with INTEL and PPO-LwD runs")
    parser.add_argument("--fused", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    graph_ids = {f"idx_{index}" for index in range(1000)}
    with (args.root / "graph_nodes.jsonl").open(encoding="utf-8") as stream:
        node_maps = [json.loads(line) for line in stream]
    node_counts = {row["graph_id"]: len(row["original_node_ids"]) for row in node_maps}
    if len(node_maps) != len(graph_ids) or node_counts.keys() != graph_ids:
        raise ValueError("The author node mapping does not cover exactly the 1000 test graphs")
    expected = {(graph, str(seed), "0") for graph in graph_ids for seed in SEEDS}
    roots = dict.fromkeys(METHODS, args.root)
    if args.follow_up:
        roots.update(dict.fromkeys(("author_rlnn", "fld_eg"), args.follow_up))
    if args.learned:
        roots.update(dict.fromkeys(("diffuco", "ltft"), args.learned))
    if args.supervised:
        roots.update(dict.fromkeys(("difusco", "dgl_treesearch"), args.supervised))
    if args.intel_ppo:
        roots.update(dict.fromkeys(("intel", "ppo_lwd"), args.intel_ppo))
    arms = {method: load(root, method, expected, node_counts) for method, root in roots.items()}
    paths = {method: root / method / "per_graph.csv" for method, root in roots.items()}
    comparisons = {f"{left}_vs_{right}": compare(paths[left], paths[right])
                   for left, right in combinations(roots, 2)}
    if args.fused:
        pattern = "pave_off_*/v11_rows.csv"
        fused = read_rows(args.fused, pattern)
        if fused.keys() != expected or any(ready is None or ready > 200 for _, _, ready, _ in fused.values()):
            raise ValueError("Fused reference does not match the saved 200 ms protocol")
        arms["fused"] = summarize(fused, 200)
        arms["fused"]["wall_overruns"] = sum(wall > 200 for _, wall, _, _ in fused.values())
        for method, path in paths.items():
            comparisons[f"fused_vs_{method}"] = compare(args.fused, path, candidate_glob=pattern)

    result = json.dumps(dict(arms=arms, comparisons=comparisons), indent=2) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
