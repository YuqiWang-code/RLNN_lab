"""Label independent RB training graphs with KaMIS OnlineMIS."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import pickle
import subprocess
from tempfile import TemporaryDirectory

import numpy as np


def label(path, output, binary):
    with path.open("rb") as stream:
        graph = pickle.load(stream)
    nodes = list(graph)
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = [" ".join(str(index[v] + 1) for v in sorted(graph[u], key=index.get)) for u in nodes]
    with TemporaryDirectory(dir=output) as directory:
        instance = Path(directory) / "instance.graph"
        solution = Path(directory) / "solution"
        instance.write_text(f"{len(nodes)} {graph.number_of_edges()}\n" + "\n".join(adjacency) + "\n",
                            encoding="ascii")
        subprocess.run([str(binary), str(instance), f"--output={solution}", "--seed=0", "--time_limit=0.25"],
                       check=True, stdout=subprocess.DEVNULL)
        labels = np.loadtxt(solution, dtype=np.int8)
    if labels.size != len(nodes) or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"{path} returned an invalid label array")
    if any(labels[index[u]] and labels[index[v]] for u, v in graph.edges):
        raise ValueError(f"{path} returned a non-independent set")
    for node, selected in zip(nodes, labels):
        graph.nodes[node]["label"] = int(selected)
    with (output / path.name).open("wb") as stream:
        pickle.dump(graph, stream, protocol=4)
    return int(labels.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.source.glob("*.gpickle"))
    if not paths:
        parser.error(f"no graph files in {args.source}")
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        sizes = list(pool.map(partial(label, output=args.output, binary=args.binary.resolve()), paths))
    print(f"{len(sizes)} labels, mean size {np.mean(sizes):.4f}")


if __name__ == "__main__":
    main()
