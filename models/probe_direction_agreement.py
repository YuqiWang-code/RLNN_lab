"""Measure how much of an oracle direction a trained static prior recovered.

Sweeps ``bias = peak * rescale((1 - lam) * learned + lam * oracle)`` on the
locked validation cohort with identical search seeds, so the lambda curve reads
as "fraction of the oracle effect recovered by the learned direction".
``lam=0`` reproduces ``amp@<peak>`` and ``lam=1`` reproduces ``oracle-kamis@<peak>``.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from eval_static_prior_v2 import add_common, cohort, load_arm_policy, search_seeds
from eval_v2 import search, sync, write_run
from mis_static_prior import bias_stats, oracle_bias
from mis_v1 import adjacency
from mis_v3_search import SearchGraph, node_mask, nodes, platform_search
from mis_v3_value import load_pave_v3
from v7_runs import digest, pave, provenance


def rescale(values):
    """Divide by the largest absolute entry so the peak equals one."""
    return values / values.abs().amax().clamp_min(1e-12)


def centred(values):
    return values - values.mean(0, keepdim=True)


def ranks(values):
    return torch.argsort(torch.argsort(values)).float()


def agreement(learned, oracle):
    """Cosine and rank statistics between the learned and oracle node orders."""
    left, right = centred(learned), centred(oracle)
    cosine = float((left.flatten() @ right.flatten())
                   / (left.norm() * right.norm()).clamp_min(1e-12))
    first, second = ranks(learned[:, 0]), ranks(oracle[:, 0])
    spread = (first - first.mean()).norm() * (second - second.mean()).norm()
    spearman = float(((first - first.mean()) @ (second - second.mean())) / spread.clamp_min(1e-12))
    k = max(1, int((oracle[:, 0] > 0).sum()))
    overlap = len(set(learned[:, 0].topk(k).indices.tolist())
                  & set(oracle[:, 0].topk(k).indices.tolist())) / k
    return dict(cosine=cosine, spearman_rank=spearman, top_overlap=overlap, target_size=k)


@torch.no_grad()
def direction_of(args, tag):
    policy, checkpoint, record = load_arm_policy(args, Path(args.checkpoint if tag == "primary" else args.reference_checkpoint))
    adapter = policy.accelerator.unwrap_model(policy.model).adapter
    if not getattr(adapter, "centered", False):
        raise ValueError(f"{tag} checkpoint has no node direction to compare")
    return policy, adapter, checkpoint, record


@torch.no_grad()
def run(args):
    dataset, graph_ids = cohort(args)
    policy, adapter, checkpoint, record = direction_of(args, "primary")
    reference_policy = reference_adapter = reference_path = None
    if args.reference_checkpoint is not None:
        reference_policy, reference_adapter, reference_path, _ = direction_of(args, "reference")
    device = policy.accelerator.device
    targets = json.loads(args.oracle_kamis_path.read_text(encoding="utf-8"))
    missing = [gid for gid in graph_ids if gid not in targets]
    if missing:
        raise ValueError(f"{args.oracle_kamis_path} lacks {missing[0]}")
    ranker = load_pave_v3(pave(args.ref))
    rows = []
    agreement_rows = []
    for index, graph in enumerate(dataset, start=args.offset):
        gid = graph_ids[index - args.offset]
        first_seed = search_seeds(index, 1)[0]
        checker = SearchGraph(graph.num_nodes, graph.edge_index, seed=first_seed)
        target = targets[gid]
        if not checker.feasible(node_mask(target)):
            raise ValueError(f"Conflicting oracle target on {gid}")
        graph = graph.to(device)
        adj = adjacency(graph)
        policy.static_intervention = None
        topology = policy.policy_topology(graph, adj)
        learned = centred(adapter.direction(topology))
        oracle = oracle_bias(graph.num_nodes, target, 1., device)
        stats = agreement(learned, oracle)
        if reference_policy is not None:
            reference_policy.static_intervention = None
            reference = centred(reference_adapter.direction(reference_policy.policy_topology(graph, adj)))
            stats.update(reference_cosine=agreement(learned, reference)["cosine"])
        agreement_rows.append(stats)
        biases = {}
        for lam in args.lambdas:
            biases[lam] = args.peak * rescale((1 - lam) * rescale(learned) + lam * oracle)
        if abs(args.lambdas[0]) < 1e-9:
            assert torch.allclose(biases[args.lambdas[0]].abs().amax(),
                                  torch.tensor(args.peak, device=device), atol=1e-5)
        if abs(args.lambdas[-1] - 1) < 1e-9:
            assert torch.allclose(biases[args.lambdas[-1]],
                                  oracle_bias(graph.num_nodes, target, args.peak, device), atol=1e-6)
        for lam in args.lambdas:
            mixed = biases[lam]

            def intervention(_adapter, _topology, _graph, mixed=mixed):
                return mixed

            policy.static_intervention = intervention
            policy.num_t = args.steps
            for seed in search_seeds(index, args.seeds_per_graph):
                legal = SearchGraph(graph.num_nodes, graph.edge_index, seed=seed)
                selected, used = search(policy, graph, legal, seed)
                if not legal.feasible(selected):
                    raise ValueError(f"Infeasible search result on {gid}")
                improved = platform_search(legal, selected, choose=ranker.choose, horizon=1).best
                if not legal.feasible(improved):
                    raise ValueError(f"Infeasible PAVE result on {gid}")
                rows.append(dict(graph_id=gid, search_seed=seed, lam=lam, steps=used,
                                 bias_rms=bias_stats(mixed)["bias_rms"],
                                 node_bias_rms=bias_stats(mixed)["node_bias_rms"],
                                 base_size=selected.bit_count(), final_size=improved.bit_count(),
                                 selected_nodes=" ".join(map(str, nodes(improved))), **stats))
        sync(device)
        if (index - args.offset + 1) % 10 == 0 or index - args.offset + 1 == len(dataset):
            print(f"lambda probe {index - args.offset + 1}/{len(dataset)}", flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per_graph.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for lam in args.lambdas:
        sizes = np.array([row["final_size"] for row in rows if row["lam"] == lam], dtype=float)
        summary[str(lam)] = dict(mean_size=float(sizes.mean()), size_sd=float(sizes.std(ddof=1)))
    summary["agreement"] = {key: float(np.mean([row[key] for row in agreement_rows]))
                            for key in ("cosine", "spearman_rank", "top_overlap", "target_size")}
    if reference_policy is not None:
        summary["agreement"]["reference_cosine"] = float(np.mean([row["reference_cosine"]
                                                                  for row in agreement_rows]))
    write_run(args.output, dict(**provenance(args.ref), version="direction_probe", status="completed",
                                backend=args.backend, checkpoint=digest(checkpoint), peak=args.peak,
                                steps=args.steps, seeds_per_graph=args.seeds_per_graph,
                                lambdas=args.lambdas, graph_ids=graph_ids,
                                oracle_sha256=digest(args.oracle_kamis_path), summary=summary,
                                reference_checkpoint=(digest(reference_path)
                                                      if reference_path else None),
                                train_arm=record.get("arm"), train_version=record.get("version")))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--peak", type=float, default=.5)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0., .25, .5, .75, 1.])
    parser.add_argument("--reference-checkpoint", type=Path,
                        help="Second adapter; reports cosine(learned, reference) to detect drift")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.oracle_kamis_path is None:
        parser.error("--oracle-kamis-path is required")
    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    if not args.lambdas or min(args.lambdas) < 0 or max(args.lambdas) > 1:
        parser.error("Use lambdas within [0, 1]")
    if not (math.isfinite(args.peak) and args.peak > 0) or args.steps < 1 or args.seeds_per_graph < 1:
        parser.error("Use a positive finite peak, step count and seed count")
    run(args)


if __name__ == "__main__":
    main()
