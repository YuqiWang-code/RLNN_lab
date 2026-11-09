"""Paired fixed-work timing of B and SC-RLNN on one idle GPU."""

import argparse
import csv
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from eval_v2 import ROOT, VALID, dataset_at, load_policy, policy_parser, search, sync, write_run
from main import set_seed
from mis_v3_search import SearchGraph
from mis_v6_frequency import V6SCGCN


@torch.no_grad()
def profile(args):
    with TemporaryDirectory() as initial:
        checkpoint = args.checkpoint or Path(initial)
        if args.checkpoint is None:
            set_seed(42)
            model = V6SCGCN(128, 5, 1.02, "v6_full")
            model.reset_parameters()
            torch.save(model.state_dict(), checkpoint / "best.pt")
        parser = policy_parser()
        policies = {}
        for label, mode, path in (
            ("B", "none", ROOT / f"outputs/v2/none_seed{args.reference_seed}/checkpoints"),
            ("SC-RLNN", "v6_full", checkpoint),
        ):
            options = parser.parse_args(["--result_dir", str(args.output), "--split", "validation",
                "--save_dir", str(path), "--frequency", mode, "--mixed_precision", args.precision])
            policies[label] = load_policy(options)
            policies[label].cuda_graph_search = args.cuda_graph
            if args.fused and mode == "v6_full":
                policies[label].accelerator.unwrap_model(policies[label].model).compile_pointwise()
    device = policies["B"].accelerator.device
    dataset = dataset_at(VALID, policies["B"].dtype, args.samples)
    for policy in policies.values():
        graph = dataset[0]
        search(policy, graph.to(device), SearchGraph(graph.num_nodes, dataset[0].edge_index, seed=1017), 1017)
    rows = []
    for index, cpu_graph in enumerate(dataset):
        seed = 1017 + index
        gid = Path(dataset.files[index]).stem
        legal = SearchGraph(cpu_graph.num_nodes, cpu_graph.edge_index, seed=seed)
        graph = cpu_graph.to(device)
        for repeat in range(args.repeats):
            order = list(policies) if (repeat + index) % 2 == 0 else list(reversed(policies))
            for label in order:
                checkpoints = {}

                def observe(step, _state, _best, _adj, _topology):
                    if step in (0, 100):
                        sync(device)
                        checkpoints[step] = perf_counter()

                sync(device)
                start = perf_counter()
                solution, steps = search(policies[label], graph, legal, seed, observe=observe)
                sync(device)
                stop = perf_counter()
                rows.append(dict(graph_id=gid, repeat=repeat, arm=label, steps=steps,
                                 search_s=stop-start, prepare_s=checkpoints[0]-start,
                                 rollout_s=checkpoints[100]-checkpoints[0],
                                 decode_s=stop-checkpoints[100], size=solution.bit_count()))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per_graph.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    times = {label: mean(row["search_s"] for row in rows if row["arm"] == label) for label in policies}
    ratio = times["SC-RLNN"] / times["B"]
    components = {label: {part: mean(row[part] for row in rows if row["arm"] == label)
                          for part in ("prepare_s", "rollout_s", "decode_s")} for label in policies}
    record = dict(status="completed", split="validation", samples=len(dataset), repeats=args.repeats,
                  checkpoint=args.checkpoint, reference_seed=args.reference_seed, precision=args.precision,
                  cuda_graph=args.cuda_graph, fused=args.fused,
                  device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                  mean_search_s=times, components=components, ratio=ratio, limit=1.08, pass_speed=ratio <= 1.08)
    write_run(args.output, record)
    print(record, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("no", "fp16"), required=True)
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    profile(parser.parse_args())
