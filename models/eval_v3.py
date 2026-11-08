"""Paired fixed-work and deadline evaluation of legal MIS plateau search."""
import csv
from datetime import datetime, timezone
from math import inf
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v2 import dataset_at, load_policy, policy_parser, search, sync, write_run
from mis_v3_search import SearchGraph, nodes, platform_search
from mis_v3_value import PAVEV3


POLICIES = ("none", "rule23", "random", "degree", "coverage", "pave")


def build_parser():
    parser = policy_parser()
    parser.set_defaults(candidate_cap=32)
    parser.add_argument("--post", nargs="+", default=["none"],
                        help="Rule policies or names registered with --extra-value-checkpoint")
    parser.add_argument("--label", required=True)
    parser.add_argument("--value-checkpoint", "--value_checkpoint", type=Path)
    parser.add_argument("--extra-value-checkpoint", "--extra_value_checkpoint", nargs=2,
                        action="append", default=[], metavar=("NAME", "PATH"))
    parser.add_argument("--horizon", type=int, choices=(1, 2), default=2)
    parser.add_argument("--outer-rounds", "--outer_rounds", type=int, default=8)
    parser.add_argument("--repair-rounds", "--repair_rounds", type=int, default=4)
    parser.add_argument("--improving-cap", "--improving_cap", type=int, default=48)
    parser.add_argument("--anchored", action="store_true")
    parser.add_argument("--time-budget-s", "--time_budget_s", type=float)
    parser.add_argument("--post-reserve-s", "--post_reserve_s", type=float, default=0.05)
    parser.add_argument("--extended-search", "--extended_search", action="store_true")
    parser.add_argument("--cohort-id", "--cohort_id")
    parser.add_argument("--backbone-seed", "--backbone_seed", type=int)
    parser.add_argument("--head-seed", "--head_seed", type=int)
    parser.add_argument("--baseline-run", "--baseline_run")
    parser.add_argument("--rule-reference-run", "--rule_reference_run")
    return parser


@torch.no_grad()
def evaluate(args):
    policies = list(dict.fromkeys(args.post))
    checkpoint_paths = {}
    for name, path in args.extra_value_checkpoint:
        if name in POLICIES or name in checkpoint_paths:
            raise ValueError(f"Duplicate or reserved extra value policy: {name}")
        checkpoint_paths[name] = Path(path)
    if args.value_checkpoint is not None:
        checkpoint_paths["pave"] = args.value_checkpoint
    unknown = set(policies) - set(POLICIES) - set(checkpoint_paths)
    if unknown:
        raise ValueError(f"Unknown post policies: {', '.join(sorted(unknown))}")
    if args.time_budget_s is not None and len(policies) != 1:
        raise ValueError("Deadline evaluation uses exactly one post policy")
    if args.extended_search and args.time_budget_s is None:
        raise ValueError("Extended search requires a time budget")
    if args.time_budget_s is not None and args.time_budget_s <= 0:
        raise ValueError("The time budget must be positive")
    if args.post_reserve_s < 0:
        raise ValueError("The post reserve cannot be negative")
    if "pave" in policies and args.value_checkpoint is None:
        raise ValueError("PAVE requires --value-checkpoint")
    solver = load_policy(args)
    solver.model.requires_grad_(False)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples, official=args.split == "test")
    if len(dataset) == 0:
        raise ValueError("Evaluation requires at least one graph")
    backbone_seed = args.train_seed if args.backbone_seed is None else args.backbone_seed
    rankers, head_seeds, head_params = {}, {}, {}
    for policy in policies:
        if policy not in checkpoint_paths:
            continue
        checkpoint = torch.load(checkpoint_paths[policy], map_location="cpu", weights_only=False)
        ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
        ranker.load_state_dict(checkpoint["model_state"])
        rankers[policy] = ranker.cpu().eval()
        head_seeds[policy] = (args.head_seed if policy == "pave" and args.head_seed is not None
                              else checkpoint.get("seed"))
        head_params[policy] = sum(parameter.numel() for parameter in ranker.parameters())
    backbone_params = sum(parameter.numel() for parameter in solver.model.parameters())
    args.result_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()

    warm = dataset[0]
    topology = SearchGraph(warm.num_nodes, warm.edge_index, seed=args.seed)
    warm_steps = solver.num_t
    solver.num_t = min(warm_steps, 100)
    search(solver, warm.to(device), topology, args.seed)
    solver.num_t = 10000 if args.extended_search else warm_steps
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    records = {policy: [] for policy in policies}
    for index, graph in enumerate(dataset):
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        graph = graph.to(device)
        sync(device)
        tick = perf_counter()
        deadline = tick + args.time_budget_s if args.time_budget_s is not None else None
        topology = SearchGraph(graph.num_nodes, edges, seed=args.seed + index)
        initial_incumbent = 1 << topology.greedy_order[0] if graph.num_nodes else 0
        cutoff = deadline
        if cutoff is not None and policies[0] != "none":
            cutoff -= args.post_reserve_s
        selected, steps = search(solver, graph, topology, args.seed + index, deadline=cutoff)
        sync(device)
        fallback_used = selected == 0 and graph.num_nodes > 0
        if fallback_used:
            selected = initial_incumbent
        shared_time = perf_counter() - tick
        for policy in policies:
            post_start = perf_counter()
            score_seconds = 0.0
            ranker = rankers.get(policy)

            def score(cpu_graph, current, actions, remaining):
                nonlocal score_seconds
                score_started = perf_counter()
                action = ranker.choose(cpu_graph, current, actions, remaining)
                score_seconds += perf_counter() - score_started
                return action

            result = None
            final = selected
            if policy != "none":
                result = platform_search(
                    topology, selected, choose=score if ranker is not None else policy,
                    outer_rounds=0 if policy == "rule23" else args.outer_rounds,
                    initial_rounds=args.exchange_rounds, horizon=args.horizon,
                    candidate_cap=args.candidate_cap, improving_cap=args.improving_cap,
                    repair_rounds=args.repair_rounds, anchored=args.anchored,
                    deadline=deadline if deadline is not None else inf,
                )
                final = result.best
            if not topology.feasible(final):
                raise ValueError(f"Invalid independent set for {graph_id}, post={policy}")
            if graph.num_nodes and not final:
                raise ValueError(f"Empty incumbent for {graph_id}, post={policy}")
            post_seconds = perf_counter() - post_start
            seconds = perf_counter() - tick if deadline is not None else shared_time + post_seconds
            trace = result.trace if result is not None else []
            size = final.bit_count()
            records[policy].append(dict(
                graph_id=graph_id, group=index // 500 + 1, backbone_seed=backbone_seed,
                head_seed=head_seeds.get(policy), search_seed=args.seed + index,
                independent_set_size=size, final_size=size, base_size=selected.bit_count(),
                rule_size=result.rule_size if result else selected.bit_count(),
                solve_time_s=seconds, search_time_s=shared_time, post_time_s=post_seconds,
                exchange_time_s=post_seconds if policy != "none" else 0.0,
                score_time_s=score_seconds, conflicts=0, search_steps=steps,
                neutral_decisions=result.neutral_decisions if result else 0,
                improving_exchanges=result.exchanges if result else 0,
                exchanges=result.exchanges if result else 0,
                outer_rounds=result.outer_steps if result else 0,
                candidate_count=sum(row.get("candidate_count", 0) for row in trace),
                full_candidate_count=sum(row.get("total_count", 0) for row in trace),
                returned_to_start=sum(row.get("returned_to_start", False) for row in trace),
                anchored_rollbacks=sum(row.get("rolled_back", False) for row in trace),
                incumbent_fallback=int(fallback_used),
                post_deadline_hit=int(result.deadline_hit) if result else 0,
                deadline_overrun=int(deadline is not None and seconds > args.time_budget_s),
                selected_nodes=" ".join(map(str, nodes(final))),
            ))
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            summary = ", ".join(f"{policy}={mean(row['final_size'] for row in rows):.4f}"
                                for policy, rows in records.items())
            print(f"{index + 1}/{len(dataset)} graphs: {summary}", flush=True)

    peak_memory = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0
    finished_at = datetime.now(timezone.utc).isoformat()
    for policy, rows in records.items():
        ranker = rankers.get(policy)
        directory = args.result_dir / policy
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "per_graph.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = dict(
            version="v3", status="completed", mode="eval", run=directory.parent.name + "/" + policy,
            method=args.label, post=policy,
            train_seed=backbone_seed, backbone_seed=backbone_seed,
            head_seed=head_seeds.get(policy), search_seed=args.seed,
            split=args.split, samples=len(rows), config=vars(args),
            baseline_run=args.baseline_run, rule_reference_run=args.rule_reference_run,
            protocol="fixed_time" if args.time_budget_s is not None else "fixed_work",
            extended_search=args.extended_search, effective_num_t=solver.num_t,
            started_at=started_at, finished_at=finished_at, wall_time_s=perf_counter() - started,
            device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
            params=backbone_params + head_params.get(policy, 0),
            trainable_params=head_params.get(policy, 0),
            value_mode=ranker.mode if ranker is not None else None,
            value_checkpoint=str(checkpoint_paths[policy]) if ranker is not None else None,
            peak_memory_mib=peak_memory,
            mean_size=mean(row["final_size"] for row in rows),
            mean_solve_time_s=mean(row["solve_time_s"] for row in rows),
            mean_post_time_s=mean(row["post_time_s"] for row in rows),
            mean_score_time_s=mean(row["score_time_s"] for row in rows),
            feasible_rate=mean(row["conflicts"] == 0 for row in rows),
            exchange_gain=mean(row["final_size"] - row["base_size"] for row in rows),
            platform_gain=mean(row["final_size"] - row["rule_size"] for row in rows),
            deadline_overruns=sum(row["deadline_overrun"] for row in rows),
            incumbent_fallbacks=sum(row["incumbent_fallback"] for row in rows),
            group_times_s=[sum(row["solve_time_s"] for row in rows if row["group"] == group)
                           for group in sorted({row["group"] for row in rows})],
            timing="Excludes dataset loading and initial H2D; includes topology, search, decode, post, scoring and feasibility",
            candidate_count_definition="Sum of retained neutral candidates at outer decision states",
        )
        if args.cohort_id:
            metadata['cohort_id'] = args.cohort_id + '/' + policy
        write_run(directory, metadata)
    return records


def main():
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
