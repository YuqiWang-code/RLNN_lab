"""Measure v5 FP16 component costs and independent end-to-end latency."""
import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from archived_frequency.diagnose_v4_branch_replay import csv_file, shared_noise
from archived_frequency.diagnose_v5_frequency import ARMS, weights
from eval_v2 import ROOT, TRAIN, dataset_at, load_policy, search, sync, write_run
from archived_frequency.eval_v5_frequency import build_parser as evaluation_parser
from mis_v1 import adjacency, best_decoded
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import load_pave_v3
from archived_frequency.mis_v5_frequency import v5_topology
from archived_frequency.profile_v4_frequency import benchmark, latency_summary


def profile_policies(seed, samples, output):
    policies = {}
    for arm in ARMS:
        options = evaluation_parser().parse_args(["--frequency", "none" if arm == "baseline" else arm,
            "--save_dir", str(weights(arm, seed).parent), "--label", arm, "--result_dir", str(output),
            "--test_path", str(TRAIN), "--test_samples", str(samples), "--mixed_precision", "fp16"])
        policy = load_policy(options)
        policy.model.requires_grad_(False)
        policies[arm] = policy
    return policies


def final_archive(policy, graph, legal, seed):
    archive = None

    def capture(step, state, best, adj, topology):
        nonlocal archive
        if step == policy.num_t:
            archive = best

    selected, _ = search(policy, graph, legal, seed, observe=capture)
    return selected, archive


@torch.no_grad()
def profile(args):
    device = torch.device("cuda")
    dataset = dataset_at(TRAIN, torch.float16, args.samples)
    policies = profile_policies(args.seed, args.samples, args.output)
    ranker = load_pave_v3(ROOT / f"outputs/v3/stats_h1_seed{args.seed}/checkpoints/best.pt").requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=False)
    e2e, components = [], []
    for index, cpu_graph in enumerate(dataset):
        gid = Path(dataset.files[index]).stem
        seed = 1017 + index
        edges = cpu_graph.edge_index
        graph = cpu_graph.to(device)
        adj = adjacency(graph)
        topo = v5_topology(graph.edge_index, adj, graph.num_nodes)
        x, _, _ = shared_noise(graph.num_nodes, 20, 1, device, graph.b.dtype, seed)
        for repeat in range(args.repeats):
            for arm in ARMS[repeat % len(ARMS):] + ARMS[:repeat % len(ARMS)]:
                policy = policies[arm]
                if index == 0 and repeat == 0:
                    search(policy, graph, SearchGraph(graph.num_nodes, edges, seed=seed), seed)
                sync(device)
                tick = perf_counter()
                legal = SearchGraph(graph.num_nodes, edges, seed=seed)
                selected, steps = search(policy, graph, legal, seed)
                sync(device)
                search_s = perf_counter()-tick
                tick = perf_counter()
                final = platform_search(legal, selected, choose=ranker.choose, horizon=1).best
                if not legal.feasible(selected) or not legal.feasible(final):
                    raise ValueError(f"Infeasible {gid}/{arm}")
                post_s = perf_counter()-tick
                e2e.append(dict(graph_id=gid, arm=arm, repeat=repeat, steps=steps, base_size=selected.bit_count(),
                    final_size=final.bit_count(), search_time_s=search_s, post_time_s=post_s, total_time_s=search_s+post_s))
        legal = SearchGraph(graph.num_nodes, edges, seed=seed)
        for arm, policy in policies.items():
            model = policy.accelerator.unwrap_model(policy.model)
            logits = policy.flip_logits(x, graph, adj, policy.policy_topology(graph, adj))
            initial_energy = policy.problem.energy_func(adj, graph.b, x)[0]
            incumbent, archive = final_archive(policy, graph, legal, seed)

            def update_archive():
                state = torch.where(torch.rand_like(x) < logits.sigmoid(), 1-x, x)
                energy = policy.problem.energy_func(adj, graph.b, state)[0]
                return torch.where((energy < initial_energy)[None], state, x)

            def base():
                with torch.autocast("cuda", dtype=torch.float16):
                    return model(x.unsqueeze(-1), graph.edge_index, topo.normalized) if arm == "baseline" else model.backbone_logits(x, graph.edge_index, topo)

            jobs = {
                "gcn_logits": (base, 100),
                "bernoulli_energy_archive": (update_archive, 100),
                "a3": (lambda: best_decoded(policy.problem, archive, adj, graph.b), 1),
                "pave": (lambda: platform_search(legal, incumbent, choose=ranker.choose, horizon=1), 1),
                "policy_topology": (lambda: policy.policy_topology(graph, adj), 1),
            }
            if arm != "baseline":
                adapter = model.adapter
                xf = x.float()
                counts = topo.binary @ xf
                _, _, gated = adapter.encode(adapter.features(xf, counts, topo))
                correction = adapter(x, topo)
                base_logits = base()
                jobs.update({
                    "Ax": (lambda: topo.binary @ xf, 100),
                    "features_encode_gate": (lambda: adapter.encode(adapter.features(xf, counts, topo)), 100),
                    "combined_projection": (lambda: adapter.bands(gated, topo), 100),
                    "logit_add_sigmoid": (lambda: (base_logits + correction.to(base_logits.dtype)).float().sigmoid(), 100),
                    "adapter_total": (lambda: adapter(x, topo), 100),
                })
            for component, (fn, factor) in jobs.items():
                values = benchmark(fn, device, args.repeats, 20 if factor == 100 else 1)
                components.append(dict(graph_id=gid, arm=arm, component=component, steps=factor,
                                       **latency_summary([value * factor for value in values])))
        print(f"{index+1}/{len(dataset)} profile graphs", flush=True)
    csv_file(args.output / "end_to_end.csv", e2e)
    csv_file(args.output / "runtime_micro.csv", components)
    write_run(args.output, dict(status="completed", split="train", seed=args.seed, precision="fp16",
        graph_ids=[Path(p).stem for p in dataset.files], repeats=args.repeats, K=20, T=100,
        mean_search_s={arm: mean(row["search_time_s"] for row in e2e if row["arm"] == arm) for arm in ARMS}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    profile(parser.parse_args())
