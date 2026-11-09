"""Inspect frequency representations on the unchanged RLNN search trajectory."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v2 import ROOT, dataset_at, search, sync, write_run
from eval_v4_frequency import build_parser as evaluation_parser, configure, prepare
from mis_v3_search import SearchGraph, nodes, platform_search
from mis_v3_value import PAVEV3
from mis_v4_frequency import state_bands, v4_topology
from models import GCN


PHASES = (0, 10, 25, 50, 75, 100)
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def csv_file(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def distribution(values):
    values = values.detach().float().cpu().flatten()
    result = dict(count=values.numel(), mean=values.mean().item(),
                  std=values.std(unbiased=False).item(), mean_abs=values.abs().mean().item())
    result.update({f"q{int(q * 100):02}": value for q, value in
                   zip(QUANTILES, torch.quantile(values, torch.tensor(QUANTILES)).tolist())})
    return result


def groups(x, counts, degree):
    degree = degree.float().cpu().flatten()
    lower, upper = torch.quantile(degree, torch.tensor([1 / 3, 2 / 3]))
    degree = degree[:, None].expand_as(x)
    masks = dict(all=torch.ones_like(x, dtype=torch.bool), selected=x.bool(),
                 unselected=~x.bool(), blockers0=counts == 0, blockers1=counts == 1,
                 blockers2plus=counts >= 2, selected_conflicted=x.bool() & (counts > 0),
                 degree_low=degree <= lower, degree_mid=(degree > lower) & (degree <= upper),
                 degree_high=degree > upper)
    for selected in (0, 1):
        for count_name, count_mask in (("0", counts == 0), ("1", counts == 1), ("2plus", counts >= 2)):
            masks[f"selected{selected}_blockers{count_name}"] = (x == selected) & count_mask
    return {name: mask for name, mask in masks.items() if mask.any()}


def spectral_basis(edge_index, size):
    binary = torch.zeros((size, size), dtype=torch.float64)
    edges = edge_index.cpu()
    binary[edges[0], edges[1]] = 1
    degree = binary.sum(1)
    inverse = degree.clamp_min(1).rsqrt()
    laplacian = torch.diag((degree > 0).double()) - inverse[:, None] * binary * inverse[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    return eigenvalues.clamp(0, 2), eigenvectors, degree.clamp_min(1).sqrt()


def spectral_statistics(signal, basis):
    """Use sqrt(D) weighting to express the lazy random walk in the L_sym basis."""
    eigenvalues, eigenvectors, weight = basis
    values = signal.detach().to(device="cpu", dtype=torch.float64).flatten(1)
    coefficients = eigenvectors.T @ (weight[:, None] * values)
    energy = coefficients.square().sum(1)
    total = energy.sum()
    denominator = total.clamp_min(1e-30)
    stats = dict(weighted_energy=total.item(),
                 spectral_centroid=(energy * eigenvalues).sum().div(denominator).item())
    for low, high in ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2.000001)):
        key = f"energy_fraction_{low:g}_{min(high, 2):g}"
        stats[key] = energy[(eigenvalues >= low) & (eigenvalues < high)].sum().div(denominator).item()
    return stats


def build_parser():
    parser = evaluation_parser()
    parser.set_defaults(split="validation", test_samples=100)
    parser.add_argument("--diagnostic-particles", type=int, default=4,
                        help="Particles used for representation and conditional distributions")
    parser.add_argument("--spectral-graphs", type=int, default=20,
                        help="First graphs with a full L_sym eigendecomposition")
    parser.add_argument("--reference-checkpoint", type=Path,
                        help="Original RLNN best.pt; defaults to the same-seed v2 none checkpoint")
    return parser


@torch.no_grad()
def diagnose(args):
    configure(args)
    if args.time_budget_s is not None or args.extended_search:
        raise ValueError("Mechanism diagnostics use the fixed K20 T100 trajectory")
    if set(args.post) - {"none", "pave"}:
        raise ValueError("Mechanism diagnostics record none and the frozen PAVE head")
    if not 1 <= args.diagnostic_particles <= args.num_k or args.spectral_graphs < 0:
        raise ValueError("Choose 1..20 diagnostic particles and a nonnegative spectral graph count")
    solver = prepare(args)
    solver.model.requires_grad_(False)
    model = solver.accelerator.unwrap_model(solver.model)
    adapter = getattr(model, "adapter", None)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples, official=args.split == "test")
    backbone_seed = args.train_seed if args.backbone_seed is None else args.backbone_seed
    reference_path = args.reference_checkpoint or ROOT / f"outputs/v2/none_seed{backbone_seed}/checkpoints/best.pt"
    reference = GCN(1, args.num_h, args.num_l, 1).to(device).eval()
    reference.load_state_dict(torch.load(reference_path, map_location=device, weights_only=True))
    reference.requires_grad_(False)
    ranker = None
    if "pave" in args.post:
        checkpoint = torch.load(args.value_checkpoint, map_location="cpu", weights_only=False)
        ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24)).cpu().eval()
        ranker.load_state_dict(checkpoint["model_state"])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    curves, conditioned, bands, spectra, eigen_rows, final_rows, timings = [], [], [], [], [], [], []
    histograms = {}
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    for index, item in enumerate(dataset):
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = item.edge_index
        cpu_graph = SearchGraph(item.num_nodes, edges, seed=args.seed + index)
        basis = spectral_basis(edges, item.num_nodes) if index < args.spectral_graphs else None
        if basis is not None:
            eigen_rows.append(dict(graph_id=graph_id, **distribution(basis[0])))
        graph = item.to(device)
        sync(device)
        graph_started = perf_counter()
        observer_seconds = 0.0

        def observe(step, state, best, adj, topology):
            nonlocal observer_seconds
            if step not in PHASES:
                return
            sync(device)
            observation_started = perf_counter()
            ft = topology if adapter is not None else v4_topology(graph.edge_index, adj, graph.num_nodes)
            raw_energy, _ = solver.problem.energy_func(adj, graph.b, state)
            best_energy, _ = solver.problem.energy_func(adj, graph.b, best)
            counts = ft.binary @ state.float()
            best_counts = ft.binary @ best.float()
            logits = solver.flip_logits(state, graph, adj, topology)
            probabilities = logits.sigmoid()
            sync(device)
            decode_start = perf_counter()
            decoded = solver.problem.decode_gradient_flip(best, adj, graph.b, return_solutions=True)
            sync(device)
            decode_seconds = perf_counter() - decode_start
            for particle in range(state.shape[1]):
                curves.append(dict(
                    graph_id=graph_id, step=step, particle=particle,
                    trajectory_elapsed_excluding_observers_s=observation_started - graph_started - observer_seconds,
                    raw_energy=raw_energy[particle].item(), best_energy=best_energy[particle].item(),
                    raw_selected=state[:, particle].sum().item(), best_selected=best[:, particle].sum().item(),
                    raw_conflict_edges=(state[:, particle].float() * counts[:, particle]).sum().item() / 2,
                    best_conflict_edges=(best[:, particle].float() * best_counts[:, particle]).sum().item() / 2,
                    decoded_best_energy_size=int(decoded[:, particle].sum().item()),
                    expected_flips=probabilities[:, particle].sum().item(),
                    expected_flip_target=args.num_d,
                ))

            particles = args.diagnostic_particles
            x = state[:, :particles].contiguous()
            sync(device)
            adapter_start = perf_counter()
            with solver.accelerator.autocast():
                details = adapter.inspect(x, adj, ft) if adapter is not None else state_bands(x, ft, args.beta)
                raw_hidden = model.initial_project(x.unsqueeze(-1))
                delta_low = details.get("delta_low", torch.zeros_like(raw_hidden))
                delta_high = details.get("delta_high", torch.zeros_like(raw_hidden))
                fused = raw_hidden + delta_low.to(raw_hidden.dtype) + delta_high.to(raw_hidden.dtype)
            sync(device)
            representation_seconds = perf_counter() - adapter_start
            with solver.accelerator.autocast():
                raw_all = model.initial_project(state.unsqueeze(-1))
                disabled_logits = model.project_hidden(raw_all, ft.normalized).squeeze(-1)
                reference_logits = reference(state.unsqueeze(-1), graph.edge_index, ft.normalized).squeeze(-1)
            actual_logits = logits[:, :particles].float()
            adapter_delta = actual_logits - disabled_logits[:, :particles].float()
            reference_delta = actual_logits - reference_logits[:, :particles].float()
            metrics = dict(logit_delta_adapter=adapter_delta, logit_delta_reference=reference_delta,
                           probability=probabilities[:, :particles],
                           probability_delta_adapter=probabilities[:, :particles] - disabled_logits[:, :particles].float().sigmoid())
            if adapter is not None:
                metrics.update(gate_low=details["gates"][..., 0], gate_high=details["gates"][..., 1])
            masks = groups(x.float().cpu(), details["c"].cpu(), ft.degree)
            for metric, values in metrics.items():
                values = values.detach().float().cpu()
                for group, mask in masks.items():
                    sample = values[mask]
                    conditioned.append(dict(graph_id=graph_id, step=step, group=group,
                                            metric=metric, **distribution(sample)))
                    if metric.startswith("gate_"):
                        key = (step, group, metric)
                        histogram = torch.histc(sample, bins=10, min=0, max=1).long()
                        histograms[key] = histograms.get(key, torch.zeros(10, dtype=torch.long)) + histogram

            z = details["z"].float()
            error = z - details["L2"].float() - details["B1"].float() - details["B0"].float()
            raw_norm = raw_hidden.float().norm().clamp_min(1e-30)
            signals = {key: details[key] for key in ("z", "L2", "B0", "B1")}
            signals.update(raw_hidden=raw_hidden, fused_hidden=fused, logits=actual_logits,
                           disabled_logits=disabled_logits[:, :particles],
                           logit_delta_adapter=adapter_delta, logit_delta_reference=reference_delta)
            if adapter is not None:
                signals.update({key: details[key] for key in ("low_input", "high_input", "u_low", "u_high",
                                                               "delta_low", "delta_high")})
            band_row = dict(graph_id=graph_id, step=step, particles=particles,
                            reconstruction_abs=error.norm().item(),
                            reconstruction_relative=(error.norm() / z.norm().clamp_min(1e-30)).item(),
                            residual_low_relative=(delta_low.float().norm() / raw_norm).item(),
                            residual_high_relative=(delta_high.float().norm() / raw_norm).item(),
                            residual_total_relative=((delta_low.float() + delta_high.float()).norm() / raw_norm).item(),
                            alpha_low=adapter.alpha_low.item() if adapter is not None else 0,
                            alpha_high=adapter.alpha_high.item() if adapter is not None else 0)
            for name, signal in signals.items():
                band_row[name + "_norm"] = signal.float().norm().item()
                band_row[name + "_rms"] = signal.float().square().mean().sqrt().item()
                if basis is not None:
                    spectra.append(dict(graph_id=graph_id, step=step, signal=name,
                                        **spectral_statistics(signal, basis)))
            bands.append(band_row)
            sync(device)
            elapsed = perf_counter() - observation_started
            observer_seconds += elapsed
            timings.append(dict(graph_id=graph_id, step=step, diagnostic_time_s=elapsed,
                                representation_time_s=representation_seconds, decode_time_s=decode_seconds))

        selected, steps = search(solver, graph, cpu_graph, args.seed + index, observe=observe)
        sync(device)
        trajectory_seconds = perf_counter() - graph_started
        if not cpu_graph.feasible(selected) or (graph.num_nodes and not selected):
            raise ValueError(f"Invalid diagnostic independent set: {graph_id}")
        row = dict(graph_id=graph_id, base_size=selected.bit_count(), final_size=selected.bit_count(),
                   rule_size=selected.bit_count(), pave_gain=0, pave_time_s=0.0, score_time_s=0.0,
                   search_steps=steps, trajectory_with_diagnostics_s=trajectory_seconds,
                   diagnostic_observer_s=observer_seconds, selected_nodes=" ".join(map(str, nodes(selected))))
        if ranker is not None:
            score_seconds = 0.0

            def choose(local_graph, current, actions, remaining):
                nonlocal score_seconds
                start = perf_counter()
                action = ranker.choose(local_graph, current, actions, remaining)
                score_seconds += perf_counter() - start
                return action

            start = perf_counter()
            result = platform_search(cpu_graph, selected, choose=choose, horizon=1, outer_rounds=8,
                                     initial_rounds=3, candidate_cap=32, improving_cap=48, repair_rounds=4)
            if not cpu_graph.feasible(result.best):
                raise ValueError(f"Invalid PAVE diagnostic independent set: {graph_id}")
            row.update(final_size=result.best.bit_count(), rule_size=result.rule_size,
                       pave_gain=result.best.bit_count() - selected.bit_count(),
                       pave_time_s=perf_counter() - start, score_time_s=score_seconds,
                       selected_nodes=" ".join(map(str, nodes(result.best))))
        final_rows.append(row)
        if (index + 1) % 10 == 0 or index + 1 == len(dataset):
            print(f"{index + 1}/{len(dataset)} graphs: base={mean(r['base_size'] for r in final_rows):.4f}, "
                  f"final={mean(r['final_size'] for r in final_rows):.4f}", flush=True)

    for filename, rows in (("energy_conflict_curves.csv", curves), ("conditional_distributions.csv", conditioned),
                           ("bands.csv", bands), ("spectra.csv", spectra), ("eigenvalues.csv", eigen_rows),
                           ("per_graph.csv", final_rows), ("diagnostic_timings.csv", timings)):
        csv_file(args.result_dir / filename, rows)
    histogram_rows = [dict(step=step, group=group, metric=metric, counts=counts.tolist())
                      for (step, group, metric), counts in sorted(histograms.items())]
    (args.result_dir / "gate_histograms.json").write_text(json.dumps(dict(
        bin_edges=[value / 10 for value in range(11)], distributions=histogram_rows), indent=2) + "\n", encoding="utf-8")
    write_run(args.result_dir, dict(
        version="v4_frequency", mode="diagnostic", status="completed", frequency=args.frequency,
        config=vars(args), train_seed=backbone_seed, search_seed=args.seed, split=args.split,
        samples=len(final_rows), diagnostic_particles=args.diagnostic_particles,
        spectral_graphs=len(eigen_rows), phases=PHASES,
        reference_checkpoint=str(reference_path),
        value_checkpoint=str(args.value_checkpoint) if ranker is not None else None,
        mean_size=mean(row["base_size"] for row in final_rows),
        mean_post_size=mean(row["final_size"] for row in final_rows),
        mean_pave_gain=mean(row["pave_gain"] for row in final_rows),
        maximum_reconstruction_relative=max(row["reconstruction_relative"] for row in bands),
        started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
        wall_time_s=perf_counter() - started,
        curve_definition="Each particle's original best-energy state, independently decoded at observation; no added archive",
        logit_delta_adapter="Current model minus the same model with both adapter residuals disabled, at the same state",
        logit_delta_reference="Current model minus original RLNN reference checkpoint, at the same state",
        spectral_definition="L_sym Fourier energy of sqrt(max(degree,1))*signal; isolated L_sym rows are zero",
        distribution_definition="Per-graph quantiles on the first diagnostic_particles; degree terciles are graph-specific",
        timing="Diagnostic observers add computation and are excluded from official solve-time comparisons",
    ))
    return final_rows


if __name__ == "__main__":
    diagnose(build_parser().parse_args())
