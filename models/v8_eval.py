"""Evaluate one v8 macro kernel against the frozen baseline.

Two protocols, both ending in the same frozen PAVE-Stats H1 with ``horizon=1``:

* ``fixed_nfe``  both arms get ``steps`` forward passes (the student one macro step
  per pass, the baseline one micro step per pass), so it isolates "value per call";
* ``budget``     both arms run to the same wall-clock deadline, which is the
  deployment criterion.

Archive rule (same for every arm): the per-particle record keeps the lowest-energy
state seen, and a candidate is only accepted after its legality is confirmed
before the deadline, so a timeout can never overwrite a better legal incumbent.

    python -u models/v8_eval.py --arm outputs/v8_macro/train_A5_ref0 --protocol budget \
        --budget_s 1.0 --reference_seed 0 --result_dir outputs/v8_macro --seeds 1017 1019 1021
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v2 import dataset_at, load_policy, sync, write_run
from mis_v1 import adjacency, best_decoded
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import PAVEV3
from mis_v8_macro import MacroKernel, energy, sample_macro
from v7_runs import digest, pave, provenance


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True, help="trained v8 arm directory")
    parser.add_argument("--reference_seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1017, 1019, 1021])
    parser.add_argument("--protocol", choices=("fixed_nfe", "budget"), default="budget")
    parser.add_argument("--steps", type=int, default=50, help="forward passes in fixed_nfe mode")
    parser.add_argument("--budget_s", type=float, default=1.0)
    parser.add_argument("--post_reserve_s", type=float, default=0.05)
    parser.add_argument("--decode_every", type=int, default=5, help="macro steps per A.3 decode")
    parser.add_argument("--num_k", type=int, default=20)
    parser.add_argument("--mixed_precision", default="no", choices=("no", "bf16", "fp16"))
    parser.add_argument("--label", default="v8_macro")
    return parser


def load_arm(path, device):
    payload = torch.load(Path(path) / "best.pt", map_location="cpu", weights_only=False)
    kernel = MacroKernel(payload["components"]).to(device)
    kernel.load_state_dict(payload["state_dict"])
    kernel.eval()
    return kernel, payload


@torch.no_grad()
def run_macro(kernel, graph, adj, generator, *, steps, deadline, decode_every, problem,
              particles, normalized):
    """Macro-step search with the lowest-energy archive and deadline-safe incumbents."""
    state = torch.randint(0, 2, (graph.num_nodes, particles), generator=generator,
                          device=graph.b.device, dtype=torch.float32)
    product = adj @ state
    best_states = state.clone()
    best_energy = energy(state, product, graph.b)
    incumbent = 0
    steps_done = 0
    for step in range(1, steps + 1):
        if deadline is not None and perf_counter() >= deadline:
            break
        y, b, _ = sample_macro(kernel, state, graph.edge_index, normalized, generator=generator)
        for candidate in (y, b) if b is not None else (y,):
            value = energy(candidate, adj @ candidate, graph.b)
            improves = value < best_energy
            best_states = torch.where(improves.unsqueeze(0), candidate, best_states)
            best_energy = torch.where(improves, value, best_energy)
        state = y
        steps_done = step
        if step % decode_every == 0:
            masks = problem.decode_gradient_flip(best_states, adj, graph.b, return_solutions=True)
            candidate = masks[:, masks.sum(dim=0).argmax()]
            integer = int.from_bytes(torch.packbits(candidate.to(torch.uint8).cpu(),
                                                    bitorder="little").numpy().tobytes(), "little")
            if integer.bit_count() > incumbent.bit_count():
                incumbent = integer
    return incumbent, steps_done


def main():
    args = build_parser().parse_args()
    args.test_path = None
    args.method = "RLNN"
    args.num_l, args.num_h, args.num_d, args.num_tp = 5, 128, 5, 50
    args.beta, args.epochs, args.batch_size, args.graph_batch_size = 1.02, 1, 1, 8
    args.lr, args.lambd, args.seed = 5e-4, 0.5, args.seeds[0]
    solver = load_policy(args)
    solver.model.requires_grad_(False)
    device = solver.accelerator.device
    kernel, payload = load_arm(args.arm, device)
    kernel.components = payload["components"]
    checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
    ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
    ranker.load_state_dict(checkpoint["model_state"])
    ranker = ranker.cpu().eval()
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / (f"{args.protocol}_" + (f"b{args.budget_s:g}" if
                                                      args.protocol == "budget" else f"nfe{args.steps}"))
    output.mkdir(exist_ok=True)
    rows = []
    started = perf_counter()
    for index, graph in enumerate(dataset):
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        graph = graph.to(device)
        adj = adjacency(graph)
        normalized = solver.policy_topology(graph, adj)
        for seed in args.seeds:
            topology = SearchGraph(graph.num_nodes, edges, seed=seed)
            for arm in ("baseline", "student"):
                sync(device)
                tick = perf_counter()
                deadline = tick + args.budget_s if args.protocol == "budget" else None
                generator = torch.Generator(device=device).manual_seed(seed)
                if arm == "student":
                    selected, steps = run_macro(
                        kernel, graph, adj, generator,
                        steps=args.steps if args.protocol == "fixed_nfe" else 10 ** 6,
                        deadline=deadline - args.post_reserve_s if deadline else None,
                        decode_every=args.decode_every, problem=solver.problem,
                        particles=args.num_k, normalized=normalized)
                else:
                    selected, steps = run_baseline(solver, graph, adj, topology, generator,
                                                   steps=args.steps if args.protocol == "fixed_nfe" else 10 ** 6,
                                                   deadline=deadline - args.post_reserve_s if deadline else None)
                final = platform_search(topology, selected, choose=ranker.choose,
                                        outer_rounds=8, initial_rounds=3, horizon=1,
                                        candidate_cap=32, improving_cap=48, repair_rounds=4,
                                        anchored=False,
                                        deadline=deadline if deadline else float("inf")).best
                sync(device)
                seconds = perf_counter() - tick
                rows.append(dict(graph_id=graph_id, search_seed=seed, arm=arm,
                                 final_size=final.bit_count(), search_size=selected.bit_count(),
                                 steps=steps, solve_time_s=round(seconds, 4),
                                 overrun=int(deadline is not None and seconds > args.budget_s)))
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"{index + 1}/{len(dataset)} " + " ".join(
                    f"{arm}={mean(r['final_size'] for r in rows if r['arm'] == arm):.4f}"
                    for arm in ("baseline", "student")), flush=True)
    path = output / f"eval_ref{args.reference_seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_run(output, dict(version="v8_macro", mode=args.protocol, ref=args.reference_seed,
                           arm=str(args.arm), horizon=1, protocol=args.protocol,
                           budget_s=args.budget_s, steps=args.steps,
                           wall_time_s=round(perf_counter() - started, 2),
                           checkpoint_sha256=digest(args.arm / "best.pt"),
                           config=vars(args), **provenance(args.reference_seed)))
    print(json.dumps({arm: mean(r["final_size"] for r in rows if r["arm"] == arm)
                      for arm in ("baseline", "student")}), flush=True)


def run_baseline(solver, graph, adj, topology, generator, *, steps, deadline):
    """Frozen micro-step search with the same archive and periodic A.3 decode."""
    problem = solver.problem
    states = solver.sample_states(graph, adj, generator=generator, normalized=topology)
    best_states = next(states)
    best_energy = energy(best_states, adj @ best_states, graph.b)
    incumbent, done = 0, 0
    for step in range(1, steps + 1):
        if deadline is not None and perf_counter() >= deadline:
            break
        state = next(states)
        value = energy(state, adj @ state, graph.b)
        improves = value < best_energy
        best_states = torch.where(improves.unsqueeze(0), state, best_states)
        best_energy = torch.where(improves, value, best_energy)
        done = step
        if step % (5 * 1) == 0:
            masks = problem.decode_gradient_flip(best_states, adj, graph.b, return_solutions=True)
            candidate = masks[:, masks.sum(dim=0).argmax()]
            integer = int.from_bytes(torch.packbits(candidate.to(torch.uint8).cpu(),
                                                    bitorder="little").numpy().tobytes(), "little")
            if integer.bit_count() > incumbent.bit_count():
                incumbent = integer
    return incumbent, done


if __name__ == "__main__":
    main()
