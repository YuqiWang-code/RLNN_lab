"""Build independently seeded MIS graphs and multi-solution supervision."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models/data"))
from xu_util import get_random_instance


REGIMES = {"RB_small": (200, 300, 0), "RB_large": (800, 1200, 10000),
           "ER_small": (700, 800, 20000)}
SEED_BASE = {"train": 4100000, "validation": 4200000}
TEACHER_SEEDS = (283, 977, 1297, 1601, 2027, 2741, 3089, 3371)


def make_graph(task):
    regime, split, index, directory = task
    lo, hi, offset = REGIMES[regime]
    seed = SEED_BASE[split] + offset + index
    random.seed(seed)
    np.random.seed(seed)
    if regime == "ER_small":
        n = random.randint(lo, hi)
        graph = nx.fast_gnp_random_graph(n, 0.15, seed=seed)
    else:
        while True:
            graph, _ = get_random_instance(regime.split("_")[1])
            graph.remove_nodes_from(list(nx.isolates(graph)))
            if lo <= len(graph) <= hi:
                break
    nodes = list(graph.nodes())
    np.random.default_rng(seed + 8000000).shuffle(nodes)
    adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, dtype=np.uint8, format="csr")
    adjacency.sort_indices()
    u, v = adjacency.nonzero()
    tensor = f"graph_{index:06d}.pt"
    torch.save({"edge_index": torch.from_numpy(np.stack((u, v))).long(),
                "num_nodes": len(nodes)}, directory / tensor)
    return dict(source=f"idx_{index}.gpickle", graph_id=f"idx_{index}", tensor=tensor,
                num_nodes=len(nodes), undirected_edges=len(u)//2, generation_seed=seed)


def make_labels(task):
    directory, item, binary, seconds = task
    graph = torch.load(directory / item["tensor"], map_location="cpu", weights_only=True)
    edge = graph["edge_index"].numpy()
    n = graph["num_nodes"]
    matrix = csr_matrix((np.ones(edge.shape[1], dtype=np.int8), edge), shape=(n, n))
    values = []
    with TemporaryDirectory() as workspace:
        path = Path(workspace)/"input.graph"
        sol = Path(workspace)/"solution"
        with path.open("w") as out:
            print(n, matrix.nnz//2, file=out)
            for node in range(n):
                row = matrix.indices[matrix.indptr[node]:matrix.indptr[node+1]]
                print(" ".join(map(str, row+1)), file=out)
        for seed in TEACHER_SEEDS:
            subprocess.run([str(binary), str(path), f"--seed={seed}", f"--time_limit={seconds}",
                            f"--output={sol}"], stdout=subprocess.DEVNULL, check=True)
            mask = np.loadtxt(sol, dtype=np.uint8)
            if len(mask) != n or mask.max() > 1 or (mask[edge[0]] & mask[edge[1]]).any():
                raise ValueError(f"Invalid teacher certificate for {directory}/{item['tensor']}")
            values.append(mask)
    label_file = directory/(item["tensor"].removesuffix(".pt")+".targets.npz")
    np.savez_compressed(label_file, masks=np.stack(values))
    return [int(mask.sum()) for mask in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--train", type=int, default=128)
    parser.add_argument("--validation", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--teacher-seconds", type=float, default=0.5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for split, count in (("validation", args.validation), ("train", args.train)):
            for regime in REGIMES:
                folder = args.output/regime/split
                folder.mkdir(parents=True, exist_ok=True)
                manifest = folder/"tensor_manifest.json"
                if manifest.exists():
                    records = json.loads(manifest.read_text())["graphs"]
                else:
                    records = list(executor.map(make_graph, ((regime, split, i, folder) for i in range(count))))
                    manifest.write_text(json.dumps(dict(count=count, regime=regime, split=split,
                        generator="xu_util.get_random_instance" if regime.startswith("RB") else "G(n,0.15)",
                        graphs=records), indent=2)+"\n")
                print(f"{regime} {split}: {len(records)} graphs ready", flush=True)
        for split in ("validation", "train"):
            for regime in REGIMES:
                folder = args.output/regime/split
                manifest = json.loads((folder/"tensor_manifest.json").read_text())
                jobs = ((folder, item, args.binary.resolve(), args.teacher_seconds) for item in manifest["graphs"])
                sizes = list(executor.map(make_labels, jobs))
                (folder/"teacher.json").write_text(json.dumps(dict(method="KaMIS OnlineMIS",
                    seconds_per_seed=args.teacher_seconds, seeds=TEACHER_SEEDS,
                    graph_sizes=sizes), indent=2)+"\n")
                print(f"{regime} {split}: {len(sizes)} labelled graphs, mean best {np.max(sizes,axis=1).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
