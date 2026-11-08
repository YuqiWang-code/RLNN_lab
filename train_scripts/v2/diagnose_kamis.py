"""Compare KaMIS with the RLNN validation solutions used in the exchange census."""
import argparse
import csv
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))
from eval_v2 import VALID, dataset_at, write_run
from mis_baselines import kamis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    with args.baseline.open(encoding="utf-8") as stream:
        baseline = {row["graph_id"]: int(row["independent_set_size"]) for row in csv.DictReader(stream)}
    dataset = dataset_at(VALID, torch.float32, len(baseline))
    rows = []
    for index, graph in enumerate(dataset):
        started = perf_counter()
        solution = kamis(graph, args.binary.resolve(), 1017 + index, 1.0)
        seconds = perf_counter() - started
        src, dst = graph.edge_index
        if (solution[src] & solution[dst]).any():
            raise ValueError("KaMIS returned a conflicting independent set")
        graph_id = Path(dataset.files[index]).stem
        size = int(solution.sum())
        rows.append(dict(graph_id=graph_id, size=size, rlnn_size=baseline[graph_id], time_s=seconds))
        if (index + 1) % 10 == 0:
            print(f"{index + 1}/{len(dataset)} graphs", flush=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    with (args.result_dir / "per_graph.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_run(args.result_dir, dict(version="v2", mode="kamis_validation", status="completed", samples=len(rows),
                                    mean_size=mean(row["size"] for row in rows),
                                    mean_delta=mean(row["size"] - row["rlnn_size"] for row in rows),
                                    mean_solve_time_s=mean(row["time_s"] for row in rows), configured_time_s=1))


if __name__ == "__main__":
    main()
