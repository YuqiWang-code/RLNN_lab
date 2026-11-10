"""Measure exchange information, residual scale and paired probe outcomes."""
import csv
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from eval_v2 import dataset_at, load_policy, policy_parser, search, sync, write_run
from mis_v1 import best_decoded
from mis_v2_exchange import ExchangeGraph, node_mask
from archived_frequency.mis_v2_frequency import exchange_band, exchange_state, frequency_topology, lazy_walk


def auc(labels, scores):
    labels = np.concatenate(labels)
    scores = np.concatenate(scores)
    positive = labels.sum()
    if 0 < positive < len(labels):
        rank_sum = rankdata(scores)[labels].sum()
        return float((rank_sum - positive * (positive + 1) / 2) / (positive * (len(labels) - positive)))
    return None


def csv_file(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


@torch.no_grad()
def diagnose(args):
    solver = load_policy(args)
    device = solver.accelerator.device
    model = solver.accelerator.unwrap_model(solver.model)
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples)
    probes = {0, args.num_t // 4, args.num_t // 2, 3 * args.num_t // 4, args.num_t}
    curves, interventions = [], []
    reports = {step: [] for step in sorted(probes)}
    samples = {step: {"labels": [], "ordinary": [], "exchange": [], "probability": []} for step in probes}
    histograms = {step: torch.zeros(2, 10) for step in probes}
    args.result_dir.mkdir(parents=True, exist_ok=True)

    for index, item in enumerate(dataset):
        local_graph = ExchangeGraph(item.num_nodes, item.edge_index)
        graph = item.to(device)
        elapsed = 0.0
        incumbent = 0
        previous = perf_counter()

        def observe(step, state, best, adj, topology):
            nonlocal elapsed, previous, incumbent
            if step not in probes:
                return
            decoded = best_decoded(solver.problem, best, adj, graph.b)
            decoded_set = node_mask(decoded.nonzero().flatten().tolist())
            sync(device)
            elapsed += perf_counter() - previous
            incumbent = max(incumbent, decoded_set.bit_count())
            curves.append(dict(graph_id=index, step=step, time_s=elapsed, best_size=incumbent))
            x = state[:, :4].contiguous()
            ft = topology if solver.segf else frequency_topology(graph.edge_index, adj, graph.num_nodes)
            features, release, released = exchange_state(x, adj, ft)
            ordinary = (features - lazy_walk(features, adj, ft)).abs().sum(-1)
            exchange = exchange_band(features, x, release, released, adj).abs().sum(-1)
            probability = solver.flip_logits(x, graph, adj, topology).sigmoid()
            labels = np.zeros(x.shape, dtype=bool)
            compatible = 0
            legal = 0
            for particle, column in enumerate(x.T.bool().tolist()):
                selected = node_mask(node for node, present in enumerate(column) if present)
                actions = local_graph.actions(selected)
                compatible += len(actions)
                if local_graph.feasible(selected):
                    legal += len(actions)
                for action in actions:
                    labels[list(action.removed + action.added), particle] = True
                if step == args.num_t // 2 and index % 4 == 0 and particle == 0 and actions:
                    # Shared future Bernoulli draws isolate the effect of the forced move.
                    rng = torch.Generator(device=device).manual_seed(args.seed + index + 100_000)
                    for node in (actions[0].removed[0], actions[0].added[0]):
                        paired = x[:, particle:particle + 1].repeat(1, 2)
                        paired[node, 1] = 1 - paired[node, 1]
                        for _ in range(4):
                            logits = solver.flip_logits(paired, graph, adj, topology)
                            uniform = torch.rand((graph.num_nodes, 1), dtype=x.dtype, device=device, generator=rng)
                            paired = torch.where(uniform < logits.sigmoid(), 1 - paired, paired)
                        sizes = solver.problem.decode_gradient_flip(paired, adj, graph.b, return_solutions=True).sum(0)
                        interventions.append(dict(graph_id=index, step=step, node=node,
                                                  flip_probability=float(probability[node, particle]),
                                                  paired_gain=int(sizes[1] - sizes[0])))
            samples[step]["labels"].append(labels.ravel())
            for name, values in (("ordinary", ordinary), ("exchange", exchange), ("probability", probability)):
                samples[step][name].append(values.float().cpu().numpy().ravel())
            report = dict(relation_edges=float(released.sum(0).mean()), released=float(release.sum(0).mean()),
                          compatible_12=compatible / x.size(1), legal_12=legal / x.size(1),
                          decoded_12=len(local_graph.actions(decoded_set)))
            if solver.frequency == "segf":
                with solver.accelerator.autocast():
                    low, high, gates = model.adapter.branches(x, adj, ft)
                    raw = model.initial_project(x.unsqueeze(-1)).float().norm(dim=-1).clamp_min(1e-8)
                report.update(
                    gate_low=float(gates[..., 0].mean()), gate_high=float(gates[..., 1].mean()),
                    relative_low=float((low.float().norm(dim=-1) / raw).mean()),
                    relative_high=float((high.float().norm(dim=-1) / raw).mean()),
                )
                for branch in range(2):
                    histograms[step][branch] += torch.histc(gates[..., branch].float(), bins=10, min=0, max=1).cpu()
            reports[step].append(report)
            sync(device)
            previous = perf_counter()

        search(solver, graph, local_graph, args.seed + index, observe=observe)
        if (index + 1) % 25 == 0:
            print(f"{index + 1}/{len(dataset)} graph trajectories", flush=True)

    phases = []
    for step, rows in reports.items():
        phase = dict(step=step, **{key: mean(row[key] for row in rows) for key in rows[0]})
        for name in ("ordinary", "exchange", "probability"):
            phase[f"auc_{name}"] = auc(samples[step]["labels"], samples[step][name])
        phases.append(phase)
    paired_corr = None
    if len({row["paired_gain"] for row in interventions}) > 1:
        correlation = spearmanr([row["flip_probability"] for row in interventions],
                                [row["paired_gain"] for row in interventions]).statistic
        paired_corr = float(correlation) if np.isfinite(correlation) else None
    csv_file(args.result_dir / "phases.csv", phases)
    csv_file(args.result_dir / "curve.csv", curves)
    csv_file(args.result_dir / "paired_probes.csv", interventions)
    write_run(args.result_dir, dict(version="v2", status="completed", mode="frequency_diagnostic",
              frequency=args.frequency, config=vars(args), samples=len(dataset), paired_probes=len(interventions),
              paired_spearman=paired_corr, gate_histograms={step: bins.tolist() for step, bins in histograms.items()}
              if solver.frequency == "segf" else None,
              phases=phases))


if __name__ == "__main__":
    parser = policy_parser()
    parser.set_defaults(split="validation", test_samples=100)
    diagnose(parser.parse_args())
