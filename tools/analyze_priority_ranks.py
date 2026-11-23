#!/usr/bin/env python3
"""Measure how much the learned node score agrees with negative degree.

The feasible-search priority is p_v = z_G(-d)_v + lam * z_G(g_theta(G))_v.
This script scores graphs with the checkpoint and reports, per graph, the rank
correlation between the learned score and negative degree, the overlap of the
top-k vertices under both orderings, and the rank correlation of the shipped
blend with degree. It runs no search and never reports a solution size.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import spearmanr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))


def standardized(values):
    centered = values - values.mean()
    return centered / max(float(np.sqrt(np.mean(centered * centered))), 1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--mix", type=float, default=0.25)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    args = parser.parse_args()

    from scale_priority import PriorityScorer

    scorer = PriorityScorer.load(args.checkpoint, device="cuda:0")
    offsets = torch.tensor([0, 1, 3, 4], dtype=torch.int32, device="cuda:0")
    nodes = torch.tensor([1, 0, 2, 1], dtype=torch.int32, device="cuda:0")
    with torch.inference_mode():
        scorer.score(offsets, nodes)

    entries = json.loads((args.dataset / "tensor_manifest.json").read_text())["graphs"]
    if args.limit:
        entries = entries[:args.limit]
    rows = []
    with torch.inference_mode():
        for i, entry in enumerate(entries):
            raw = torch.load(args.dataset / entry["tensor"], weights_only=True, map_location="cpu")
            edges = raw["edge_index"].numpy()
            n = int(raw["num_nodes"])
            csr = coo_matrix((np.ones(edges.shape[1], np.int8), (edges[0], edges[1])),
                             shape=(n, n)).tocsr()
            indptr = torch.from_numpy(csr.indptr.astype(np.int32)).cuda()
            indices = torch.from_numpy(csr.indices.astype(np.int32)).cuda()
            neural = scorer.score(indptr, indices).float().cpu().numpy()
            degree = np.diff(csr.indptr).astype(np.float32)
            k = max(5, int(round(args.top_fraction * n)))
            top_neural = set(np.argsort(-neural)[:k].tolist())
            top_degree = set(np.argsort(-(-degree))[:k].tolist())
            blend = standardized(-degree) + args.mix * standardized(neural)
            rows.append(dict(
                graph_id=Path(entry["source"]).stem,
                nodes=n,
                spearman_neural_degree=float(spearmanr(neural, -degree).statistic),
                top_overlap=len(top_neural & top_degree) / k,
                spearman_blend_degree=float(spearmanr(blend, -degree).statistic),
                spearman_blend_neural=float(spearmanr(blend, neural).statistic),
            ))
            if (i + 1) % 50 == 0 or i + 1 == len(entries):
                print(f"{i+1}/{len(entries)} graphs", flush=True)

    args.output.mkdir(parents=True, exist_ok=False)
    import csv
    with (args.output / "per_graph.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: float(np.mean([row[key] for row in rows]))
               for key in ("spearman_neural_degree", "top_overlap",
                           "spearman_blend_degree", "spearman_blend_neural")}
    summary.update(graphs=len(rows), mix=args.mix, top_k_fraction=args.top_fraction)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
