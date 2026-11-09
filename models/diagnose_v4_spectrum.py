"""FP32 stage spectra on common teacher states and each student's own states."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import torch

from eval_v2 import TRAIN, VALID, dataset_at
from frequency_diag_utils import (
    BAND_BOUNDARY_TOLERANCE, node_groups, override_logits, spectral_basis,
    spectral_energy_many, trace_model,
)
from mis_v1 import adjacency
from mis_v4_frequency import V4FrequencyGCN, v4_topology
from models import GCN


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("v4a", "v4a_message", "v4a_mlp")
PHASES = (0, 10, 25, 50, 75, 100)


class CsvOutput:
    def __init__(self, path):
        self.stream = path.open("w", encoding="utf-8", newline="")
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        self.stream.close()


def load_model(mode, checkpoint, device):
    model = GCN(1, 128, 5, 1) if mode == "none" else V4FrequencyGCN(128, 5, mode=mode)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.float().to(device).eval().requires_grad_(False)


@torch.no_grad()
def trajectory_states(model, mode, graph, adj, topology, initial, uniforms, phases, particles):
    x = initial.clone()
    snapshots = {0: x[:, :particles].clone()} if 0 in phases else {}
    for step, noise in enumerate(uniforms, 1):
        if mode == "none":
            logits = model(x.unsqueeze(-1), graph.edge_index, topology.normalized).squeeze(-1)
        else:
            logits = model(x, graph.edge_index, adj, topology)
        x = torch.where(noise < logits.sigmoid(), 1 - x, x)
        if step in phases:
            snapshots[step] = x[:, :particles].clone()
    return snapshots


def rms(values):
    return float(values.float().square().mean().sqrt())


def correlation(first, second):
    first = first.double().flatten()
    second = second.double().flatten()
    if first.min() == first.max() or second.min() == second.max():
        return None
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    return float((first * second).sum() / denominator) if denominator > 1e-20 else None


@torch.no_grad()
def behavior_row(common, group, mask, details, signals, full_logits, no_high, no_adapter):
    logits = signals["logits"]
    probabilities = logits.sigmoid()
    hidden = signals["raw_hidden"][mask].float()
    low = details["delta_low"][mask].float()
    high = details["delta_high"][mask].float()
    high_change = (logits - no_high)[mask]
    normalizer = hidden.norm().clamp_min(1e-30)
    row = dict(
        **common, group=group, count=int(mask.sum()),
        selected_fraction=float(details["x"][mask].mean()),
        c_mean=float(details["c"][mask].mean()), r_mean=float(details["r"][mask].mean()),
        gate_low_mean=float(details["gates"][..., 0][mask].mean()),
        gate_high_mean=float(details["gates"][..., 1][mask].mean()),
        raw_hidden_rms=rms(hidden), delta_low_rms=rms(low), delta_high_rms=rms(high),
        residual_low_relative=float(low.norm() / normalizer),
        residual_high_relative=float(high.norm() / normalizer),
        residual_total_relative=float((low + high).norm() / normalizer),
        logit_change_from_full_rms=rms((logits - full_logits)[mask]),
        probability_change_from_full_mean_abs=float((probabilities - full_logits.sigmoid())[mask].abs().mean()),
        logit_delta_high_rms=rms(high_change),
        probability_delta_high_mean_abs=float((probabilities - no_high.sigmoid())[mask].abs().mean()),
        logit_delta_adapter_rms=rms((logits - no_adapter)[mask]),
        probability_below_001=float((probabilities[mask] < 0.01).float().mean()),
        probability_above_099=float((probabilities[mask] > 0.99).float().mean()),
        expected_flips=float(probabilities.sum(0).mean()) if group == "all" else None,
    )
    for band in ("B0", "B1"):
        for channel, label in enumerate(("x", "logc")):
            row[f"corr_{band}_{label}_high_logit_change"] = correlation(details[band][..., channel][mask], high_change)
    return row


@torch.no_grad()
def diagnose(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if not 1 <= args.particles <= 20 or any(step < 0 or step > 100 for step in args.phases):
        raise ValueError("P1 observes 1..20 particles at phases in 0..100")
    graphs = args.graphs if args.graphs is not None else (20 if args.split == "train" else 100)
    if graphs <= 0:
        raise ValueError("P1 graph count must be positive")
    data_path = args.data_path or (TRAIN if args.split == "train" else VALID)
    dataset = dataset_at(data_path, torch.float32, graphs)
    if not len(dataset):
        raise ValueError("P1 requires at least one cached graph")
    checkpoint_root = args.checkpoint_root
    teacher_path = args.teacher_checkpoint or checkpoint_root / f"outputs/v2/none_seed{args.seed}/checkpoints/best.pt"
    teacher = load_model("none", teacher_path, device)
    paths = {mode: checkpoint_root / f"outputs/v4_frequency/{mode}_seed{args.seed}/checkpoints/best.pt" for mode in args.models}
    models = {mode: load_model(mode, path, device) for mode, path in paths.items()}
    phases = sorted(set(args.phases))
    origins = ("teacher", "own") if args.trajectory == "both" else (args.trajectory,)
    graph_ids = [Path(file).stem for file in dataset.files]
    args.result_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: CsvOutput(args.result_dir / f"{name}.csv") for name in
               ("stage_spectra", "stationary_removed", "gates_and_deltas", "graph_spectral_support")}
    metadata = dict(
        version="v4_frequency_followup", task="p1_spectrum", status="running", seed=args.seed,
        config=vars(args), models=args.models, graph_ids=graph_ids, split=args.split,
        role="exploratory" if args.split == "train" else "validation_recheck",
        dtype="float32", spectral_dtype="float64_cpu", state_source=args.trajectory,
        actual_K=20, actual_T=100, diagnostic_particles=args.particles, phases=phases,
        checkpoints={**paths, "teacher": teacher_path}, postprocess="none",
        variants=["full", "high_no_bias0", "high_no_bias2"],
        band_boundaries=[0, 0.5, 1, 1.5, 2], band_boundary_tolerance=BAND_BOUNDARY_TOLERANCE,
        stationary_eigenvalue_epsilon=1e-7, centered_relative_energy_floor=1e-12,
        spectral_definition="L_sym projection of sqrt(max(degree,1))*signal; remove entire zero eigenspace",
        constant_definition="node_constant: unweighted global node mean; weighted_global_constant: degree-weighted global node mean",
        randomness="Same x0 and 100 pre-generated per-step N×20 FP32 uniforms for teacher and own trajectories; search_seed+graph_index",
        timing="Diagnostic wall time only; not solver latency", started_at=datetime.now(timezone.utc).isoformat(),
    )

    def save():
        (args.result_dir / "run.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    save()
    start = perf_counter()
    observations = spectral_records = 0
    try:
        for index, graph in enumerate(dataset):
            graph_id = graph_ids[index]
            basis = spectral_basis(graph.edge_index, graph.num_nodes)
            eigenvalues = basis[0]
            outputs["graph_spectral_support"].write(dict(
                graph_id=graph_id, seed=args.seed, nodes=graph.num_nodes,
                zero_modes=int((eigenvalues <= 1e-7).sum()),
                eigenvalue_min=float(eigenvalues.min()), eigenvalue_max=float(eigenvalues.max()),
                eigenvalue_mean=float(eigenvalues.mean()),
            ))
            graph = graph.to(device)
            adj = adjacency(graph)
            topology = v4_topology(graph.edge_index, adj, graph.num_nodes)
            generator = torch.Generator(device=device).manual_seed(args.search_seed + index)
            initial = torch.randint(0, 2, (graph.num_nodes, 20), dtype=torch.float32, device=device, generator=generator)
            uniforms = torch.stack([torch.rand(initial.shape, device=device, generator=generator) for _ in range(100)])
            teacher_states = trajectory_states(teacher, "none", graph, adj, topology, initial,
                                              uniforms, phases, args.particles) if "teacher" in origins else None
            for mode, model in models.items():
                own_states = trajectory_states(model, mode, graph, adj, topology, initial,
                                                uniforms, phases, args.particles) if "own" in origins else None
                for origin in origins:
                    snapshots = teacher_states if origin == "teacher" else own_states
                    for step in phases:
                        x = snapshots[step]
                        common = dict(graph_id=graph_id, seed=args.seed, model=mode, trajectory=origin,
                                      step=step, particles=args.particles)
                        full_details, full_signals = trace_model(model, x, graph.edge_index, adj, topology)
                        no_high = override_logits(model, x, graph.edge_index, adj, topology, high_scale=0)
                        no_adapter = model(x, graph.edge_index, adj, topology, disable_adapter=True)
                        variants = {"full": (full_details, full_signals)}
                        for layer in (0, 2):
                            variants[f"high_no_bias{layer}"] = trace_model(
                                model, x, graph.edge_index, adj, topology, zero_bias={"high": (layer,)})
                        for variant, (details, signals) in variants.items():
                            selected_signals = dict(signals) if variant == "full" else {
                                name: value for name, value in signals.items()
                                if name.startswith("high_") or name in ("delta_high", "fused_hidden", "logits")}
                            selected_signals["logit_delta_high"] = signals["logits"] - no_high
                            selected_signals["logit_delta_adapter"] = signals["logits"] - no_adapter
                            selected_signals["logit_delta_from_full"] = signals["logits"] - full_signals["logits"]
                            statistics = spectral_energy_many(selected_signals, basis)
                            for signal, record in statistics.items():
                                row = dict(**common, signal=signal, variant=variant, **record)
                                outputs["stage_spectra"].write(row)
                                centered = {key: value for key, value in record.items()
                                            if key.startswith(("centered_", "stationary_", "node_constant_", "weighted_global_constant_"))
                                            or key in ("nonstationary_energy", "valid_centered", "zero_modes", "total_energy")}
                                outputs["stationary_removed"].write(dict(**common, signal=signal, variant=variant, **centered))
                                spectral_records += 1
                            groups = node_groups(x, details["c"], details["r"]) if variant == "full" else {
                                "all": torch.ones_like(x, dtype=torch.bool)}
                            for group, mask in groups.items():
                                if mask.any():
                                    row = behavior_row(dict(**common, variant=variant), group, mask, details,
                                                       signals, full_signals["logits"], no_high, no_adapter)
                                    row.update(alpha_low=float(model.adapter.alpha_low), alpha_high=float(model.adapter.alpha_high))
                                    outputs["gates_and_deltas"].write(row)
                        observations += 1
            print(f"{index + 1}/{len(dataset)} graphs; {observations} state/model observations; {perf_counter() - start:.1f}s", flush=True)
        metadata.update(status="completed", graphs=len(dataset), observations=observations, spectral_records=spectral_records)
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        for output in outputs.values():
            output.close()
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), wall_time_s=perf_counter() - start)
        save()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--graphs", type=int)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--search-seed", type=int, default=1017)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--trajectory", choices=("teacher", "own", "both"), default="both")
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--phases", nargs="+", type=int, default=list(PHASES))
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--result-dir", type=Path, required=True)
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
