"""Generate independent ER validation graphs for the OnlineMIS executable."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(16):
        seed = 4_230_000 + index
        rng = np.random.default_rng(seed)
        n = int(rng.integers(9000, 11001))
        neighbors = [i + 1 + np.flatnonzero(rng.random(n - i - 1) < 0.02) for i in range(n)]
        u = np.repeat(np.arange(n), [len(v) for v in neighbors])
        v = np.concatenate(neighbors)
        upper = coo_matrix((np.ones(len(u), dtype=np.uint8), (u, v)), shape=(n, n)).tocsr()
        src, dst = (upper + upper.T).nonzero()
        tensor = f"graph_{index:06d}.pt"
        torch.save({"edge_index": torch.from_numpy(np.stack((src, dst))).long(),
                    "num_nodes": n}, output / tensor)
        records.append(dict(source=f"idx_{index}.gpickle", graph_id=f"idx_{index}", tensor=tensor,
                            num_nodes=n, undirected_edges=len(u), generation_seed=seed))
    manifest = dict(count=len(records), regime="ER_large", split="validation",
                    generator="Uniform integer n in [9000,11000]; independent Bernoulli(0.02) edges",
                    graphs=records)
    (output / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
