"""Measure v5 state signals, shared-noise action changes and complete MIS replay."""
import argparse
from pathlib import Path
from statistics import mean

import torch

from diagnose_v4_branch_replay import PHASES, csv_file, replay, same_state_rows, shared_noise
from eval_v2 import ROOT, TRAIN, VALID, dataset_at, write_run
from frequency_diag_utils import spectral_basis, spectral_energy_many
from mis_v1 import adjacency
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import load_pave_v3
from mis_v4_frequency import lazy_walk
from mis_v5_frequency import V5DAFRGCN, v5_topology
from models import GCN
from problems.mis import Problem
from probe_v4_utility import rank_metrics
from probe_v4_utility_followup import label_reliability, node_masks, paired_utilities, sample_candidates


ARMS = ("baseline", "v5a_mlp", "v5a_message", "v5a")


def weights(arm, seed, initial=False):
    if arm == "baseline" or initial:
        return ROOT / f"outputs/v2/none_seed{seed}/checkpoints/best.pt"
    return ROOT / f"outputs/v5_frequency/train_{arm}_seed{seed}/checkpoints/best.pt"


def load(arm, seed, device, initial=False):
    model = GCN(1, 128, 5, 1) if arm == "baseline" else V5DAFRGCN(128, 5, mode=arm)
    missing, extra = model.load_state_dict(torch.load(weights(arm, seed, initial), weights_only=True, map_location="cpu"),
                                           strict=not initial or arm == "baseline")
    if extra or any(not key.startswith("adapter.") for key in missing):
        raise ValueError(f"Incompatible checkpoint for {arm}")
    return model.to(device).eval().requires_grad_(False)


def score(model, x, edges, adj, topo, precision):
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=precision == "fp16"):
        if hasattr(model, "adapter"):
            return model(x, edges, adj, topo).float()
        return model(x.unsqueeze(-1), edges, topo.normalized).squeeze(-1).float()


@torch.no_grad()
def run(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cuda")
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    arms = ("baseline", "v5a") if args.initial else ARMS
    models = {arm: load(arm, args.seed, device, args.initial) for arm in arms}
    dataset = dataset_at(TRAIN if args.split == "train" else VALID, dtype, args.samples)
    problem = Problem(1.02)
    head_path = ROOT / f"outputs/v3/stats_h1_seed{args.seed}/checkpoints/best.pt"
    ranker = None if args.initial else load_pave_v3(head_path).requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = {key: [] for key in ("reconstruction", "stage_spectra", "node_groups", "gate_summary", "replay", "replay_curves",
                               "native_repeat", "utility_ranking", "utility_reliability")}
    for index, cpu_graph in enumerate(dataset):
        gid = Path(dataset.files[index]).stem
        graph_seed = 1017 + index
        basis = spectral_basis(cpu_graph.edge_index, cpu_graph.num_nodes)
        legal = SearchGraph(cpu_graph.num_nodes, cpu_graph.edge_index, seed=graph_seed)
        graph = cpu_graph.to(device)
        adj = adjacency(graph)
        topo = v5_topology(graph.edge_index, adj, graph.num_nodes)
        initial, uniforms, terminal = shared_noise(graph.num_nodes, 20, 100, device, dtype, graph_seed)
        results = {}
        for arm, model in models.items():
            def forward(x):
                return score(model, x, graph.edge_index, adj, topo, args.precision)
            result = replay(problem, adj, graph.b, initial, uniforms, forward, capture=True)
            results[arm] = result
            start = result["selected"]
            final = (platform_search(legal, start, choose=ranker.choose, horizon=1).best
                     if ranker is not None else start)
            if not legal.feasible(start) or not legal.feasible(final):
                raise ValueError(f"Infeasible replay at {gid}/{arm}")
            common = dict(graph_id=gid, seed=args.seed, arm=arm, precision=args.precision)
            counts = topo.binary @ result["best_states"].float()
            conflicts = (counts * result["best_states"].float()).sum(0) / 2
            rows["replay"].append(dict(**common, base_size=start.bit_count(), final_size=final.bit_count(),
                                       pave_gain=final.bit_count()-start.bit_count(),
                                       best_conflicts_mean=float(conflicts.mean()), feasible=1))
            rows["replay_curves"].extend(dict(**common, **row) for row in result["curves"])
            if index < args.repeat_graphs and arm in ("baseline", "v5a"):
                repeat = replay(problem, adj, graph.b, initial, uniforms, forward, capture=True)
                rows["native_repeat"].append(dict(**common, size_delta=repeat["selected"].bit_count()-start.bit_count(),
                    changed_phases=sum(not torch.equal(result["states"][t], repeat["states"][t]) for t in PHASES)))
        for step in PHASES:
            teacher_x = results["baseline"]["states"][step][:, :4]
            scores = {arm: score(model, teacher_x, graph.edge_index, adj, topo, args.precision)
                      for arm, model in models.items()}
            scores["full"] = scores.pop("v5a")
            noise = (uniforms[step] if step < 100 else terminal)[:, :4]
            counts = topo.binary @ teacher_x.float()
            release = teacher_x.float() * (topo.binary @ ((teacher_x == 0) & (counts == 1)).float())
            rows["node_groups"].extend(dict(graph_id=gid, seed=args.seed, step=step, **row)
                for row in same_state_rows(teacher_x, counts, release, scores, noise, binary=topo.binary))
            if args.utility and step in (25, 50, 100):
                state = teacher_x[:, 0].float()
                masks, _ = node_masks(state, counts[:, 0], release[:, 0])
                for sampling, count in (("uniform", 16), ("stratified", 24)):
                    candidate_seed = (graph_seed + 10000 * args.seed) * 101 + step
                    candidates = sample_candidates(masks, count, sampling, candidate_seed).to(device)
                    _, gains = paired_utilities(models["baseline"], state, candidates, graph, adj, topo.normalized,
                                                problem, 20, candidate_seed, replicates=8)
                    gains = gains.cpu()
                    info = dict(graph_id=gid, seed=args.seed, step=step, sampling=sampling, replicates=8, horizon=20)
                    rows["utility_reliability"].append(dict(**info, **label_reliability(gains)))
                    for arm, logit in scores.items():
                        metrics = rank_metrics(logit[candidates, 0].cpu(), gains.mean(0))
                        rows["utility_ranking"].append(dict(**info, arm=arm, **metrics))
            for arm in arms[1:]:
                model = models[arm]
                for source, x in (("teacher", teacher_x), ("own", results[arm]["states"][step][:, :4])):
                    corr, details = model.adapter(x, topo, return_details=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=args.precision == "fp16"):
                        base = model.backbone_logits(x, graph.edge_index, topo)
                        final = base + corr.to(base.dtype)
                    common = dict(graph_id=gid, seed=args.seed, step=step, arm=arm, trajectory=source)
                    signals = {key: details[key] for key in ("z", "proposal", "gated", "low", "high")}
                    signals.update(correction=corr, logits_base=base, logits_final=final,
                                   delta_probability=final.float().sigmoid()-base.float().sigmoid())
                    stats = spectral_energy_many(signals, basis)
                    rows["stage_spectra"].extend(dict(**common, signal=name, **value) for name, value in stats.items())
                    if arm == "v5a":
                        p = lazy_walk(details["gated"], topo)
                        reconstruction = (p + (details["gated"] - p) - details["gated"]).abs()
                        rows["reconstruction"].append(dict(**common, max_abs_error=float(reconstruction.max()),
                            max_relative_error=float(reconstruction.max()/details["gated"].abs().max().clamp_min(1e-12)),
                            high_stationary_fraction=stats["high"]["stationary_fraction"]))
                    for j, name in enumerate(("low", "high")):
                        gate = details["gates"][..., j].flatten()
                        rows["gate_summary"].append(dict(**common, branch=name, mean=float(gate.mean()),
                                                         p10=float(gate.quantile(.1)), p90=float(gate.quantile(.9))))
        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(f"{index+1}/{len(dataset)} paired graphs", flush=True)
    for key, records in rows.items():
        csv_file(args.output / f"{key}.csv", records)
    write_run(args.output, dict(status="completed", split=args.split, seed=args.seed, precision=args.precision,
        initial=args.initial, samples=len(dataset), K=20, T=100, search_seed=1017,
        graph_ids=[Path(p).stem for p in dataset.files], params={k: sum(p.numel() for p in m.parameters()) for k,m in models.items()},
        base_sizes={arm: mean(row["base_size"] for row in rows["replay"] if row["arm"] == arm) for arm in arms},
        checkpoints={arm: str(weights(arm, args.seed, args.initial)) for arm in arms},
        pave_checkpoint=str(head_path) if ranker is not None else None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--initial", action="store_true")
    parser.add_argument("--repeat-graphs", type=int, default=0)
    parser.add_argument("--utility", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.utility and (args.precision != "fp32" or args.split != "train" or args.initial):
        parser.error("Utility diagnostics use trained policies on FP32 training graphs")
    run(args)
