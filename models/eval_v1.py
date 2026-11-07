"""Score solver outputs on the same ordered RLD4CO MIS graphs."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from main import build_parser, set_seed
from mis_baselines import gurobi, isco, kamis, minimum_degree
from mis_v1 import FlipValue, search
from problems.mis import Dataset, Problem
from solvers import RLNN, RLSA


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "models/data/author_rld4co/MIS/RB_small_test"


def ordered_dataset(args, dtype):
    dataset = Dataset(args.test_path, dtype)
    files = {Path(name).stem: name for name in dataset.files}
    count = 1000 if args.test_samples < 0 else args.test_samples
    dataset.files = [files[f"idx_{i}"] for i in range(count)]
    return dataset


@torch.no_grad()
def evaluate(args, solve, device, params, *, dtype=torch.float32):
    dataset = ordered_dataset(args, dtype)
    result = Path(args.result_dir)
    result.mkdir(parents=True, exist_ok=True)
    fields = ["graph_id", "group", "independent_set_size", "solve_time_s", "conflicts",
              "extra_candidates", "archive_seconds", "archive_gain", "archive_gap", "baseline_archive_gap"]
    rows = []
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with (result / "per_graph.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(dataset)):
            graph = dataset[index].to(device)
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            solution, details = solve(graph, generator, index)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = perf_counter() - start
            u, v = graph.edge_index
            mask = solution.bool()
            row = dict(graph_id=f"idx_{index}", group=index // 500 + 1,
                       independent_set_size=int(mask.sum()), solve_time_s=seconds,
                       conflicts=int((mask[u] & mask[v]).sum()) // 2, **details)
            writer.writerow(row)
            rows.append(row)
            if (index + 1) % 100 == 0:
                stream.flush()
                print(f"{index + 1}/{len(dataset)} graphs, mean size {mean(r['independent_set_size'] for r in rows):.3f}", flush=True)
    metadata = dict(
        version="v1", status="completed", mode="eval", run=result.name, method=args.label,
        train_seed=args.train_seed, search_seed=args.seed, samples=len(rows), config=vars(args),
        started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
        wall_time_s=perf_counter() - started,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), params=params,
        peak_memory_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
        mean_size=mean(row["independent_set_size"] for row in rows),
        mean_solve_time_s=mean(row["solve_time_s"] for row in rows),
        feasible_rate=mean(row["conflicts"] == 0 for row in rows),
        measurement="Single graph. Synchronized wall time includes solver setup, search and decoding. Graph loading, transfer and solution verification are outside the timer.",
    )
    (result / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("mean_size", "mean_solve_time_s", "feasible_rate", "params", "peak_memory_mib")}), flush=True)


def add_evaluation_arguments(parser):
    parser.add_argument("--label", required=True)
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--train_seed", type=int, default=0)
    return parser


def main():
    parser = add_evaluation_arguments(build_parser())
    parser.set_defaults(method="RLNN", test_path=str(TEST), mixed_precision="fp16",
                        num_k=20, num_t=100, num_l=5, num_h=128, num_d=5, num_workers=0)
    parser.add_argument("--archive", choices=["off", "quality", "energy"], default="off")
    parser.add_argument("--archive_probes", type=int, default=5)
    parser.add_argument("--archive_candidates", type=int, default=4)
    parser.add_argument("--value_checkpoint", type=Path)
    parser.add_argument("--value_energy", action="store_true")
    parser.add_argument("--value_weight", type=float, default=0.5)
    parser.add_argument("--time_budget_s", type=float)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--kamis", type=Path)
    parser.add_argument("--cpu_time_limit", type=float, default=1.0)
    args = parser.parse_args()
    set_seed(args.seed)
    params = 0
    env = None
    if args.method == "RLNN":
        solver = RLNN(args)
        solver.prepare()
        solver.model.eval()
        device, dtype = solver.accelerator.device, solver.dtype
        params = sum(p.numel() for p in solver.model.parameters())
        value = "energy" if args.value_energy else None
        if args.value_checkpoint:
            value = FlipValue().to(device).eval()
            value.load_state_dict(torch.load(args.value_checkpoint, map_location=device, weights_only=True))
            params += sum(p.numel() for p in value.parameters())

        def solve(graph, generator, index):
            return search(solver, graph, generator, archive=args.archive, probes=args.archive_probes,
                          candidates=args.archive_candidates, value=value, value_weight=args.value_weight,
                          time_budget=args.time_budget_s, diagnostic=args.diagnostic)
    elif args.method == "RLSA":
        solver = RLSA(args)
        device, dtype = solver.accelerator.device, solver.dtype

        def solve(graph, generator, index):
            return solver.evaluate_single(graph, return_solution=True, generator=generator), {}
    elif args.method == "iSCO":
        device, dtype = torch.device("cuda"), torch.float32
        problem = Problem(args.beta)

        def solve(graph, generator, index):
            return isco(problem, graph, generator, args.num_t, args.num_k, args.num_d, args.tau0), {}
    else:
        device, dtype = torch.device("cpu"), torch.float32
        if args.method == "Greedy":
            def solve(graph, generator, index):
                return minimum_degree(graph), {}
        elif args.method == "KaMIS":
            if args.kamis is None:
                parser.error("--kamis must point to redumis")

            def solve(graph, generator, index):
                return kamis(graph, args.kamis.resolve(), args.seed + index, args.cpu_time_limit), {}
        elif args.method == "Gurobi":
            import gurobipy as gp
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()

            def solve(graph, generator, index):
                return gurobi(graph, env, args.seed + index, args.cpu_time_limit), {}
        else:
            parser.error(f"unknown method {args.method}")

    try:
        evaluate(args, solve, device, params, dtype=dtype)
    finally:
        if env is not None:
            env.dispose()


if __name__ == "__main__":
    main()
