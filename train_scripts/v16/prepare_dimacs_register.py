"""Prepare the 14 REG graphs in the DIMACS coloring benchmark as MIS inputs."""
import argparse
import json
from pathlib import Path
from subprocess import run

import torch


ARCHIVE = "https://archive.dimacs.rutgers.edu/pub/challenge/graph/benchmarks/color/"
CATALOG = "https://mat.tepper.cmu.edu/COLOR/instances.html"
CASES = {"fpsol2": 3, "inithx": 3, "mulsol": 5, "zeroin": 3}


def load_graph(path):
    edges = set()
    records = 0
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if not fields or fields[0] == "c":
            continue
        if fields[:2] == ["p", "edge"]:
            num_nodes, num_records = map(int, fields[2:])
        elif fields[0] == "e":
            a, b = (int(v) - 1 for v in fields[1:])
            if not 0 <= a < num_nodes or not 0 <= b < num_nodes or a == b:
                raise ValueError(f"Invalid edge in {path}: {line}")
            records += 1
            edges.add((min(a, b), max(a, b)))
        else:
            raise ValueError(f"Unexpected DIMACS record in {path}: {line}")
    if records != num_records:
        raise ValueError(f"Header declares {num_records} edges, read {records} in {path}")
    directed = sorted(edges | {(b, a) for a, b in edges})
    return num_nodes, len(edges), torch.tensor(directed, dtype=torch.long).T.contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    entries = []
    for family, count in CASES.items():
        for index in range(1, count + 1):
            name = f"{family}.i.{index}.col"
            path = raw / name
            if not path.exists():
                run(["curl", "--fail", "--silent", "--show-error", "--location",
                     "--retry", "2", ARCHIVE + name, "--output", str(path)], check=True)
            n, m, edge_index = load_graph(path)
            tensor = path.stem + ".pt"
            torch.save(dict(num_nodes=n, edge_index=edge_index), args.output / tensor)
            entries.append(dict(source=name, tensor=tensor, num_nodes=n, num_edges=m))
    manifest = dict(catalog=CATALOG, archive=ARCHIVE, selection="All catalog entries labeled REG",
                    graphs=entries)
    (args.output / "tensor_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                       encoding="utf-8")
    print(f"{len(entries)} graphs written to {args.output}")


if __name__ == "__main__":
    main()
