"""Measure RLNN inference on the two published RB200-300 MIS groups."""

import csv
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from main import build_parser, set_seed
from solvers import RLNN


GROUP_SIZE = 500


@torch.no_grad()
def evaluate(solver, result_dir, decoder):
    solver.model.eval()
    evaluate_graph = solver.evaluate_single
    if decoder == "gradient_flip":
        evaluate_graph = partial(evaluate_graph, decode=solver.problem.decode_gradient_flip)
    device = solver.accelerator.device
    groups = []
    with (result_dir / "per_graph.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "graph_id", "independent_set_size", "solve_time_s"])
        sizes = []
        seconds = []
        for index, (graph,) in enumerate(solver.test_loader):
            torch.cuda.synchronize(device)
            start = perf_counter()
            raw_size = evaluate_graph(graph)
            torch.cuda.synchronize(device)
            elapsed = perf_counter() - start
            size = -int(raw_size.item())
            sizes.append(size)
            seconds.append(elapsed)
            group = index // GROUP_SIZE + 1
            writer.writerow([group, f"idx_{index}", size, f"{elapsed:.9f}"])

            if len(sizes) % 100 == 0:
                stream.flush()
                print(f"group {group}: {len(sizes)}/{GROUP_SIZE}, "
                      f"SIZE {sum(sizes) / len(sizes):.3f}, TIME {sum(seconds):.2f}s", flush=True)
            if len(sizes) == GROUP_SIZE:
                groups.append((sum(sizes) / GROUP_SIZE, sum(seconds)))
                sizes.clear()
                seconds.clear()
    return groups


def main():
    parser = build_parser()
    parser.set_defaults(method="RLNN", problem="mis", loss="erdoes", num_t=100,
                        num_k=20, num_l=5, num_h=128, mixed_precision="fp16")
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--decoder", choices=["greedy", "gradient_flip"], default="greedy")
    parser.add_argument("--checkpoint_epoch", type=int)
    parser.add_argument("--checkpoint_source")
    args = parser.parse_args()
    if args.decoder == "gradient_flip" and (args.problem != "mis" or args.skip_decode):
        parser.error("gradient_flip requires MIS with decoding enabled")
    set_seed(args.seed)
    solver = RLNN(args)
    dataset = solver.test_loader.dataset
    expected = [f"idx_{index}.gpickle" for index in range(2 * GROUP_SIZE)]
    by_name = {Path(file).name: file for file in dataset.files}
    if set(by_name) != set(expected):
        parser.error("test_path must contain the author files idx_0.gpickle through idx_999.gpickle")
    dataset.files = [by_name[name] for name in expected]
    solver.prepare()
    if solver.accelerator.device.type != "cuda" or solver.accelerator.num_processes != 1:
        parser.error("this benchmark requires one CUDA process")

    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now(timezone.utc).isoformat()
    groups = evaluate(solver, result_dir, args.decoder)
    (group1_size, group1_time), (group2_size, group2_time) = groups
    (result_dir / "metrics").mkdir(exist_ok=True)
    with (result_dir / "metrics/metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "group1/independent_set_size", "group1/total_time_s",
                         "group2/independent_set_size", "group2/total_time_s",
                         "test/independent_set_size", "test/total_time_s"])
        writer.writerow([args.checkpoint_epoch, group1_size, group1_time, group2_size, group2_time,
                         (group1_size + group2_size) / 2, (group1_time + group2_time) / 2])

    metadata = {
        "version": result_dir.parent.name,
        "run": result_dir.name,
        "task": "MIS",
        "dataset": "RLD4CO author RB200-300",
        "kind": "single_gpu",
        "mode": "eval",
        "status": "completed",
        "seed": args.seed,
        "started_at": start_time,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **vars(args),
            "result_dir": str(result_dir),
            "test_samples": 1000,
            "groups": ["idx_0–499", "idx_500–999"],
        },
        "device": {
            "name": torch.cuda.get_device_name(solver.accelerator.device),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "allocator_backend": torch.cuda.get_allocator_backend(),
            "cpu_threads": torch.get_num_threads(),
        },
        "measurement": (
            f"One graph at a time. Wall clock around sampling and {args.decoder} decoding, "
            "with CUDA synchronization on both sides. Includes the first graph; excludes "
            "data loading, host-to-device transfer and model startup. SIZE is the mean "
            "cardinality per graph. TIME is the mean of the two 500-graph solver totals."
        ),
        "reference": {
            "paper": "https://arxiv.org/pdf/2502.00277#page=7",
            "table": "Table 1, MIS RB-[200–300], RLNN (UL)",
            "gpu": "NVIDIA RTX A6000",
            "independent_set_size": 19.52,
            "total_time_minutes": 1.64,
            "data": (
                "https://huggingface.co/datasets/shengyuf/RLD4CO/tree/"
                "f273683b3fe6faa217de35ead96fbe0ee4a44a4a/MIS/RB_small_test"
            ),
        },
    }
    (result_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
