"""Evaluate RLNN/SEGF and post-decode exchange policies on the same RB graphs."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import torch

from main import build_parser, set_seed
from mis_v1 import adjacency, best_decoded
from mis_v2_exchange import ExchangeGraph, action_values, improve, node_mask, nodes
from mis_v2_value import PAVE
from problems.mis import Dataset
from solvers import RLNN


ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT / "models/data/MIS_tensors/RB_small_val"
TRAIN = ROOT / "models/data/MIS_tensors/RB_small_train"
TEST = ROOT / "models/data/author_rld4co/MIS/RB_small_test"
POLICIES = ("none", "rule", "heuristic", "random", "pave", "rule23")


def policy_parser():
    parser = build_parser()
    parser.set_defaults(method="RLNN", test_path=None, mixed_precision="fp16", seed=1017,
                        num_k=20, num_t=100, num_l=5, num_h=128, num_d=5, num_workers=0)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--candidate_cap", type=int, default=48)
    parser.add_argument("--exchange_rounds", type=int, default=3)
    parser.add_argument("--larger", action="store_true", help="try 2-to-3 when no 1-to-2 action exists")
    return parser


def dataset_at(path, dtype, count=-1, *, official=False):
    dataset = Dataset(path, dtype)
    if official:
        by_id = {Path(file).stem: file for file in dataset.files}
        dataset.files = [by_id[f"idx_{index}"] for index in range(len(dataset) if count < 0 else count)]
    elif count > 0:
        dataset.files = dataset.files[:count]
    return dataset


def load_policy(args):
    if args.test_path is None:
        args.test_path = str(VALID if args.split == "validation" else TEST)
    set_seed(args.seed)
    solver = RLNN(args)
    solver.prepare()
    solver.model.eval()
    return solver


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def search(solver, graph, cpu_graph, seed, *, deadline=None, observe=None):
    """Keep the original per-trajectory energy record and A.3 decoder."""
    device = graph.b.device
    generator = torch.Generator(device=device).manual_seed(seed)
    adj = adjacency(graph)
    topology = solver.policy_topology(graph, adj)
    states = solver.sample_states(graph, adj, generator=generator, normalized=topology)
    problem = solver.problem
    best_states = next(states)
    best_energy, _ = problem.energy_func(adj, graph.b, best_states)
    incumbent = 0
    chunk_time = 0.0
    chunk_start = perf_counter()

    def decode():
        selected = best_decoded(problem, best_states, adj, graph.b)
        mask = node_mask(selected.nonzero().flatten().tolist())
        if not cpu_graph.feasible(mask):
            raise ValueError("A.3 returned a conflicting independent set")
        return mask

    if deadline is not None:
        first = decode()
        if perf_counter() < deadline:
            incumbent = first
    if observe is not None:
        observe(0, best_states, best_states, adj, topology)
    steps = 0
    for step in range(1, solver.num_t + 1):
        if deadline is not None and perf_counter() + chunk_time >= deadline:
            break
        state = next(states)
        energy, _ = problem.energy_func(adj, graph.b, state)
        improves = energy < best_energy
        best_states = torch.where(improves.unsqueeze(0), state, best_states)
        best_energy = torch.where(improves, energy, best_energy)
        steps = step
        if observe is not None:
            observe(step, state, best_states, adj, topology)
        if deadline is not None and (step % 10 == 0 or step == solver.num_t):
            candidate = decode()
            now = perf_counter()
            if now >= deadline:
                break
            if candidate.bit_count() > incumbent.bit_count():
                incumbent = candidate
            chunk_time = now - chunk_start
            chunk_start = now
    return (decode() if deadline is None else incumbent), steps


def write_run(result, metadata):
    path = result / "run.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def diagnose_actions(graph, selected, cap, rounds):
    row = {}
    for name, larger in (("one_to_two", False), ("fallback", True)):
        all_actions = graph.actions(selected, larger=larger)
        actions = all_actions[:cap]
        values = action_values(graph, selected, actions, cap=cap, rounds=rounds, larger=larger)
        counts = {value: values.count(value) for value in set(values)}
        ties = sum(count * (count - 1) // 2 for count in counts.values())
        row[f"{name}_actions"] = len(all_actions)
        row[f"{name}_pairs"] = len(actions) * (len(actions) - 1) // 2
        row[f"{name}_ties"] = ties
        row[f"{name}_spread"] = max(values) - min(values) if values else 0
        row[f"{name}_oracle_gain"] = max(values, default=selected.bit_count()) - selected.bit_count()
    return row


@torch.no_grad()
def evaluate(args):
    solver = load_policy(args)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples, official=args.split == "test")
    policy_names = list(dict.fromkeys(args.post))
    if args.opportunities and "none" not in policy_names:
        raise ValueError("Opportunity diagnosis needs the original decoded solution")
    if args.time_budget_s is not None and (args.opportunities or len(policy_names) != 1):
        raise ValueError("Fixed-time runs take one post policy and no opportunity diagnosis")
    ranker = None
    if "pave" in policy_names:
        if not args.value_checkpoint:
            raise ValueError("PAVE evaluation requires --value_checkpoint")
        ranker = PAVE().eval()
        ranker.load_state_dict(torch.load(args.value_checkpoint, map_location="cpu", weights_only=True))
    model_params = sum(parameter.numel() for parameter in solver.model.parameters())
    args.result_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    warm = dataset[0]
    cpu_warm = ExchangeGraph(warm.num_nodes, warm.edge_index)
    search(solver, warm.to(device), cpu_warm, args.seed)
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records = {name: [] for name in policy_names}
    opportunities = []
    for index, graph in enumerate(dataset):
        edges = graph.edge_index
        graph = graph.to(device)
        sync(device)
        tick = perf_counter()
        deadline = tick + args.time_budget_s if args.time_budget_s is not None else None
        cutoff = deadline
        if deadline is not None and policy_names[0] != "none":
            cutoff -= args.exchange_reserve_s
        cpu_graph = ExchangeGraph(graph.num_nodes, edges)
        selected, steps = search(solver, graph, cpu_graph, args.seed + index, deadline=cutoff)
        sync(device)
        search_time = perf_counter() - tick
        for name in policy_names:
            post_start = perf_counter()
            choose = None
            if name == "pave":
                choose = ranker.choose
            elif name == "rule" or name == "rule23":
                choose = lambda graph, selected, actions: actions[0]
            elif name == "random":
                rng = random.Random(args.seed + index)
                choose = lambda graph, selected, actions: rng.choice(actions)
            final, exchanges, candidate_count = selected, 0, 0
            if name != "none":
                final, exchanges, candidate_count = improve(
                    cpu_graph, selected, cap=args.candidate_cap, rounds=args.exchange_rounds,
                    larger=name == "rule23" or (args.larger and name not in ("none", "rule")),
                    deadline=deadline if deadline is not None else float("inf"), choose=choose,
                    repeat_choice=name != "pave")
            extra_time = perf_counter() - post_start if name != "none" else 0.0
            size = final.bit_count()
            seconds = (perf_counter() - tick) if deadline is not None else search_time + extra_time
            records[name].append(dict(graph_id=Path(dataset.files[index]).stem, group=index // 500 + 1,
                                      independent_set_size=size, solve_time_s=seconds,
                                      conflicts=0, search_steps=steps, base_size=selected.bit_count(),
                                      exchanges=exchanges, exchange_candidates=candidate_count,
                                      exchange_time_s=extra_time, selected_nodes=" ".join(map(str, nodes(final)))))
        if args.opportunities:
            opportunities.append(dict(graph_id=Path(dataset.files[index]).stem, base_size=selected.bit_count(),
                                      **diagnose_actions(cpu_graph, selected, args.candidate_cap, args.exchange_rounds)))
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            scores = ", ".join(f"{name}={mean(row['independent_set_size'] for row in rows):.3f}"
                               for name, rows in records.items())
            print(f"{index + 1}/{len(dataset)} graphs: {scores}", flush=True)
    for name, rows in records.items():
        result = args.result_dir / name
        result.mkdir(exist_ok=True)
        with (result / "per_graph.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = dict(
            version="v2", status="completed", mode="eval", run=result.parent.name + "/" + name,
            method=args.label, post=name, train_seed=args.train_seed, search_seed=args.seed,
            split=args.split, samples=len(rows), config=vars(args), started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(), wall_time_s=perf_counter() - started,
            device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
            params=model_params + (sum(p.numel() for p in ranker.parameters()) if name == "pave" else 0),
            peak_memory_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
            mean_size=mean(row["independent_set_size"] for row in rows),
            mean_solve_time_s=mean(row["solve_time_s"] for row in rows), feasible_rate=1.0,
            group_times_s=[sum(row["solve_time_s"] for row in rows if row["group"] == group)
                           for group in sorted({row["group"] for row in rows})],
            exchange_gain=mean(row["independent_set_size"] - row["base_size"] for row in rows),
            deadline_overruns=sum(row["solve_time_s"] > args.time_budget_s for row in rows)
            if args.time_budget_s is not None else 0,
        )
        write_run(result, metadata)
    if opportunities:
        with (args.result_dir / "opportunities.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(opportunities[0]))
            writer.writeheader()
            writer.writerows(opportunities)
        summary = {"graphs": len(opportunities)}
        for name in ("one_to_two", "fallback"):
            summary[name] = dict(
                available=sum(row[f"{name}_actions"] > 0 for row in opportunities),
                informative=sum(row[f"{name}_spread"] > 0 for row in opportunities),
                mean_actions=mean(row[f"{name}_actions"] for row in opportunities),
                mean_oracle_gain=mean(row[f"{name}_oracle_gain"] for row in opportunities),
                pairs=sum(row[f"{name}_pairs"] for row in opportunities),
                ties=sum(row[f"{name}_ties"] for row in opportunities),
            )
        write_run(args.result_dir, dict(version="v2", mode="opportunities", status="completed", **summary))


def main():
    parser = policy_parser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--post", choices=POLICIES, nargs="+", default=["none"])
    parser.add_argument("--value_checkpoint", type=Path)
    parser.add_argument("--time_budget_s", type=float)
    parser.add_argument("--exchange_reserve_s", type=float, default=0.02)
    parser.add_argument("--opportunities", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
