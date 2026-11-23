"""Convert the labeled RB training graphs to the upstream DGL input format."""
import argparse
from pathlib import Path
import pickle

import dgl
import numpy as np
import torch


def convert(path):
    with path.open("rb") as stream:
        graph = pickle.load(stream)
    nodes = sorted(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    edges = np.asarray([(index[u], index[v]) for u, v in graph.edges], dtype=np.int64)
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
    result = dgl.graph(tuple(torch.from_numpy(directed.T)), num_nodes=len(nodes))
    result.ndata["label"] = torch.tensor([graph.nodes[node]["label"] for node in nodes])
    result.ndata["weight"] = torch.ones((len(nodes), 1))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labeled_train", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.labeled_train.glob("*.gpickle"))
    graphs = [convert(path) for path in paths]
    dgl.save_graphs(str(args.output), graphs)
    print(f"Wrote {len(graphs)} graphs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
