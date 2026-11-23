"""Compare frozen 200 ms MIS runs on the three author test splits."""

import argparse
import csv
from contextlib import nullcontext
import json
from pathlib import Path
from statistics import median
import sys

from compare_mis_runs import compare
from summarize_reference_deadline import load as validate_archive
from summarize_scale_deadline import ROOT, SEEDS, SPLITS, summarize as validate_deadline


V11 = ROOT / "outputs/v11"
SCALE = V11 / "scale_deadline_20261120"
REFERENCES = {
    "RB_small": ("OnlineMIS", "CPU, one thread", V11 / "reference_deadline_20261119/kamis_online"),
    "RB_large": ("FLD-EG (reconstruction)", "RTX 5090", SCALE / "RB_large/fld_eg"),
    "ER_small": ("RLSA", "RTX 5090", SCALE / "ER_small/rlsa"),
}


def author_node_counts(split, tensor_root):
    if tensor_root is not None:
        with (tensor_root / f"{split}_author_test/tensor_manifest.json").open(encoding="utf-8") as stream:
            return {Path(r["source"]).stem: r["num_nodes"] for r in json.load(stream)["graphs"]}
    if split == "RB_small":
        with (V11 / "reference_deadline_20261119/graph_nodes.jsonl").open(encoding="utf-8") as stream:
            records = (json.loads(line) for line in stream)
            return {r["graph_id"]: len(r["original_node_ids"]) for r in records}
    with (V11 / f"scale_author_data_20261120/{split.lower()}.json").open(encoding="utf-8") as stream:
        return {r["graph_id"]: r["num_nodes"] for r in json.load(stream)["graphs"]}


def verify_edges(directory, split, tensor_root):
    import numpy as np
    import torch

    cache = tensor_root / f"{split}_author_test"
    with (cache / "tensor_manifest.json").open(encoding="utf-8") as stream:
        graphs = {Path(r["source"]).stem: cache / r["tensor"] for r in json.load(stream)["graphs"]}
    current = None
    updates = 0
    with (directory / "solutions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            graph_id = record["graph_id"]
            if graph_id != current:
                graph = torch.load(graphs[graph_id], map_location="cpu", weights_only=True)
                adjacency = np.zeros((graph["num_nodes"], graph["num_nodes"]), dtype=bool)
                u, v = graph["edge_index"].numpy()
                adjacency[u, v] = True
                current = graph_id
            for update in record["updates"]:
                nodes = np.asarray(update["nodes"], dtype=np.int64)
                if adjacency[np.ix_(nodes, nodes)].any():
                    raise ValueError(f"Adjacent nodes in {directory}, {graph_id}, seed {record['seed']}")
                updates += 1
    return updates


def delivery_stats(path, graph_bounds):
    first_times = []
    certified_graphs = set()
    certified_pairs = 0
    with (path.parent / "solutions.jsonl").open(encoding="utf-8") as stream:
        first_events = {(r["graph_id"], int(r["seed"])): r["updates"][0]["ready_ms"] if r["updates"] else None
                        for r in map(json.loads, stream)}
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        has_bound = "upper_bound" in rows.fieldnames
        for row in rows:
            key = row["graph_id"]
            first, last = row["first_ready_ms"], row["final_ready_ms"]
            if bool(first) != bool(last) or (first and not (0 <= float(first) <= float(last))):
                raise ValueError(f"Inconsistent first and final delivery times in {path}, {key}")
            if (float(first) if first else None) != first_events[key, int(row["seed"])]:
                raise ValueError(f"First delivery differs from the archived event in {path}, {key}")
            if first:
                first_times.append(float(first))
            size = int(row["final_size"])
            lower, upper = graph_bounds[key]
            if has_bound:
                run_bound = int(row["upper_bound"])
                upper = min(upper, run_bound)
                if last and size == run_bound:
                    certified_pairs += 1
                    certified_graphs.add(key)
            lower = max(lower, size)
            if upper < lower:
                raise ValueError(f"Reported upper bound {upper} below a feasible size {lower} for {key} in {path}")
            graph_bounds[key] = lower, upper
    return dict(first_ready_p50_ms=median(first_times) if first_times else None,
                certified_pairs=certified_pairs if has_bound else None,
                certified_graphs=len(certified_graphs) if has_bound else None)


def contrast(split, name, path, ref_name, ref_device, ref_path, delivery):
    with (path.parent / "run.json").open(encoding="utf-8") as stream:
        method_meta = json.load(stream)
    with (ref_path.parent / "run.json").open(encoding="utf-8") as stream:
        reference_meta = json.load(stream)
    result = compare(path, ref_path)
    row = {"split": split, "method": name, "method_device": priority_device(method_meta),
           "reference": ref_name, "reference_device": ref_device or priority_device(reference_meta),
           "graphs": result["graph_count"], "pairs": result["pairs"]}
    for prefix, key in (("", "candidate"), ("reference_", "reference")):
        stats = result[key]
        for field in ("mean_size", "mean_wall_ms", "wall_p95_ms", "ready_by_deadline"):
            row[prefix + field] = stats[field]
        source = path if key == "candidate" else ref_path
        row.update({prefix + k: v for k, v in delivery[source].items()})
    lo, hi = result["ci95"]
    row.update(paired_gain=result["mean_paired_delta"], ci95_low=lo, ci95_high=hi,
               wins=result["wins"], ties=result["ties"], losses=result["losses"],
               method_timing=method_meta["timing"], reference_timing=reference_meta["timing"],
               method_source=archive_path(path), reference_source=archive_path(ref_path))
    return row


def archive_path(path):
    full = path.resolve()
    return (full.relative_to(ROOT) if full.is_relative_to(ROOT) else full).as_posix()


def priority_device(meta):
    if meta.get("checkpoint"):
        return "CPU + RTX 5090 (priority inference)"
    return "CPU"


def score(root, arms, tensor_root, tuned_cpu):
    results = []
    for split, (champion, device, previous) in REFERENCES.items():
        counts = author_node_counts(split, tensor_root)
        if len(counts) != SPLITS[split]:
            raise ValueError(f"Incomplete author graph inventory for {split}")
        expected = {(g, str(seed), "0") for g in counts for seed in SEEDS}
        primary_name, primary_template = arms[0]
        primary = root / primary_template.format(split=split) / "per_graph.csv"
        reference = previous / "per_graph.csv"
        validate_deadline(reference, SPLITS[split])
        tuned = tuned_cpu / split / "per_graph.csv"
        validate_deadline(tuned, SPLITS[split])
        validate_archive(tuned_cpu, split, expected, counts)
        if tensor_root:
            verify_edges(tuned.parent, split, tensor_root)
        graph_bounds = {key: (0, n) for key, n in counts.items()}
        delivery = {p: delivery_stats(p, graph_bounds) for p in (reference, tuned)}
        for name, template in arms:
            directory = root / template.format(split=split)
            path = directory / "per_graph.csv"
            validate_deadline(path, SPLITS[split])
            validate_archive(directory.parent, directory.name, expected, counts)
            if tensor_root:
                verify_edges(directory, split, tensor_root)
            delivery[path] = delivery_stats(path, graph_bounds)
            results.append(contrast(split, name, path, champion, device, reference, delivery))
            results.append(contrast(split, name, path, "OnlineMIS (validation-selected time limit)",
                                    "CPU, one pinned thread", tuned, delivery))
            if name != primary_name:
                results.append(contrast(split, primary_name, primary, name, None, path, delivery))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--arm", nargs=2, action="append", metavar=("NAME", "DIR_TEMPLATE"),
                        help="Directory below ROOT. Defaults to learned, degree and neutral; first arm is primary.")
    parser.add_argument("--verify-edges", type=Path, metavar="MIS_TENSORS",
                        help="Check every archived update against the original graph tensors on CPU.")
    parser.add_argument("--online-mis", type=Path, default=ROOT / "outputs/v15/priority_mis_20261121/online_mis_tuned/test",
                        help="Root with the three validation-selected OnlineMIS test runs")
    parser.add_argument("--output", type=Path, help="Output CSV (default: stdout)")
    args = parser.parse_args()
    arms = args.arm or [(name, "{split}/" + name) for name in ("learned", "degree", "neutral")]
    if len({name for name, _ in arms}) != len(arms):
        parser.error("arm names must be distinct")
    rows = score(args.root, arms, args.verify_edges, args.online_mis)
    destination = args.output.open("w", newline="", encoding="utf-8") if args.output else nullcontext(sys.stdout)
    with destination as stream:
        out = csv.DictWriter(stream, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


if __name__ == "__main__":
    main()
