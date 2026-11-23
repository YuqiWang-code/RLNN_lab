"""Summarize primary-cohort delivered sizes by number of vertices."""
import csv
import json
from pathlib import Path
from statistics import fmean
import sys

from compare_mis_runs import read_rows


def main():
    root = Path(__file__).resolve().parents[1] / "outputs/v11"
    reference = root / "reference_deadline_20261119"
    manifest = (reference / "graph_nodes.jsonl").read_text(encoding="utf-8")
    nodes = {r["graph_id"]: len(r["original_node_ids"])
             for r in map(json.loads, manifest.splitlines())}
    fused = read_rows(root / "fused_pave_02s_deadline", "pave_off_s*/v11_rows.csv")
    controls = [
        read_rows(root / "reference_deadline_20261119_followup/author_rlnn/per_graph.csv"),
        read_rows(reference / "rlsa/per_graph.csv"),
        read_rows(reference / "kamis_online/per_graph.csv"),
    ]
    if set(nodes) != {key[0] for key in fused} or any(c.keys() != fused.keys() for c in controls):
        raise ValueError("Node counts and solver records must cover the same cohort")
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["min_nodes", "max_nodes", "graphs", "mean_fused",
                     "delta_released", "delta_rlsa", "delta_onlinemis"])
    for lo, hi in ((200, 219), (220, 239), (240, 264)):
        keys = [k for k in fused if lo <= nodes[k[0]] <= hi]
        delta = [fmean(fused[k][0] - c[k][0] for k in keys) for c in controls]
        writer.writerow([lo, hi, len({k[0] for k in keys}), f"{fmean(fused[k][0] for k in keys):.4f}",
                         *(f"{x:+.4f}" for x in delta)])


if __name__ == "__main__":
    main()
