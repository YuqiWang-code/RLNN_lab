"""Paired branch interventions with a shared RLNN initialization and flip stream."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

from eval_v2 import ROOT, TRAIN, VALID, dataset_at, load_policy, policy_parser, sync, write_run
from archived_frequency.frequency_diag_utils import intervention_details, override_logits
from mis_v1 import adjacency
from mis_v3_search import SearchGraph, node_mask, nodes, platform_search
from mis_v3_value import PAVEV3
from archived_frequency.mis_v4_frequency import v4_topology


PHASES = (0, 10, 25, 50, 75, 100)
MODEL_MODES = dict(baseline="none", tuned="none", mlp="v4a_mlp", message="v4a_message", f4="v4a")
INTERVENTIONS = {
    "full": {}, "no_low": {"low_scale": 0.0}, "no_high": {"high_scale": 0.0},
    "gate_half": {"gate": 0.5}, "no_adapter": {"low_scale": 0.0, "high_scale": 0.0},
    "low_bias1": {"zero_bias": {"low": (0,)}}, "low_bias2": {"zero_bias": {"low": (2,)}},
    "high_bias1": {"zero_bias": {"high": (0,)}}, "high_bias2": {"zero_bias": {"high": (2,)}},
    "low_identity": {"activation": {"low": "identity"}},
    "high_identity": {"activation": {"high": "identity"}},
    "low_no_norm": {"skip_norm": ("low",)}, "high_no_norm": {"skip_norm": ("high",)},
}


def add_common_arguments(parser):
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0, help="Checkpoint training seed")
    parser.add_argument("--search-seed", type=int, default=1017)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--test-path", type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--num-k", type=int, default=20)
    parser.add_argument("--num-t", type=int, default=100)
    for name in MODEL_MODES:
        flag = "base" if name == "baseline" else name
        parser.add_argument(f"--checkpoint-{flag}", type=Path, dest=f"checkpoint_{name}")
    parser.add_argument("--result-dir", type=Path, required=True)
    return parser


def checkpoint_paths(args):
    paths = dict(baseline=ROOT / f"outputs/v2/none_seed{args.seed}/checkpoints/best.pt",
                 tuned=ROOT / f"outputs/v4_frequency/none_seed{args.seed}/checkpoints/best.pt")
    for name in ("mlp", "message", "f4"):
        paths[name] = ROOT / f"outputs/v4_frequency/{MODEL_MODES[name]}_seed{args.seed}/checkpoints/best.pt"
    for name in paths:
        supplied = getattr(args, f"checkpoint_{name}")
        if supplied is not None:
            paths[name] = supplied / "best.pt" if supplied.is_dir() else supplied
    return paths


def load_group(args, names=tuple(MODEL_MODES)):
    """Use the original solver's precision wrapper and checkpoint loader."""
    if args.num_k != 20 or args.num_t != 100:
        raise ValueError("The follow-up fixes K=20 and T=100")
    args.test_path = args.test_path or (TRAIN if args.split == "train" else VALID)
    paths, solvers = checkpoint_paths(args), {}
    for name in names:
        path = paths[name]
        if path.name != "best.pt" or not path.is_file():
            raise ValueError(f"Missing best.pt checkpoint for {name}: {path}")
        options = policy_parser().parse_args([
            "--result_dir", str(args.result_dir), "--save_dir", str(path.parent),
            "--frequency", MODEL_MODES[name], "--test_path", str(args.test_path),
            "--test_samples", str(args.graphs), "--mixed_precision", "no" if args.precision == "fp32" else "fp16",
            "--seed", str(args.search_seed), "--num_k", "20", "--num_t", "100",
        ])
        solver = load_policy(options)
        solver.model.requires_grad_(False)
        solvers[name] = solver
    return solvers, {name: str(paths[name]) for name in names}


def csv_file(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def shared_noise(size, particles, steps, device, dtype, seed):
    """Precompute the original per-step rand calls, preserving their shapes and dtype."""
    generator = torch.Generator(device=device).manual_seed(seed)
    initial = torch.randint(0, 2, (size, particles), device=device, dtype=dtype, generator=generator)
    uniforms = torch.stack([torch.rand(initial.shape, device=device, dtype=dtype, generator=generator)
                            for _ in range(steps)])
    terminal_probe = torch.rand(initial.shape, device=device, dtype=dtype, generator=generator)
    return initial, uniforms, terminal_probe


def policy_logits(solver, graph, adj, topology, state, intervention=None):
    if not intervention:
        return solver.flip_logits(state, graph, adj, topology)
    model = solver.accelerator.unwrap_model(solver.model)
    with solver.accelerator.autocast():
        return override_logits(model, state, graph.edge_index, adj, topology, **intervention).float()


@torch.no_grad()
def replay(problem, adj, bias, initial, uniforms, forward, *, phases=PHASES, capture=False):
    """Maintain one energy record per particle; decode only the final records."""
    state, best = initial.clone(), initial.clone()
    best_energy, _ = problem.energy_func(adj, bias, state)
    states, rows = {}, []

    def record(step, energy, probability=None, flips=None):
        if step not in phases:
            return
        if capture:
            states[step] = state.clone()
        counts = 2 * (adj.float() @ state.float())
        conflicts = (counts * state.float()).sum(0) / 2
        for particle in range(state.shape[1]):
            rows.append(dict(step=step, particle=particle, raw_energy=energy[particle].item(),
                             best_energy=best_energy[particle].item(), conflict_edges=conflicts[particle].item(),
                             selected_count=int(state[:, particle].sum().item()), transition_from=step - 1 if step else None,
                             expected_flips=probability[:, particle].sum().item() if probability is not None else None,
                             actual_flips=int(flips[:, particle].sum().item()) if flips is not None else None))

    record(0, best_energy)
    for step, uniform in enumerate(uniforms, 1):
        probability = forward(state).sigmoid()
        flips = uniform < probability
        state = torch.where(flips, 1 - state, state)
        energy, _ = problem.energy_func(adj, bias, state)
        better = energy < best_energy
        best = torch.where(better.unsqueeze(0), state, best)
        best_energy = torch.where(better, energy, best_energy)
        record(step, energy, probability, flips)
    decoded = problem.decode_gradient_flip(best, adj, bias, return_solutions=True)
    selected = decoded[:, decoded.float().sum(0).argmax()]
    return dict(selected=node_mask(selected.nonzero().flatten().tolist()), states=states, curves=rows,
                best_states=best, final_state=state)


def variants(args):
    result = {name: (name, {}) for name in args.models if name != "f4"}
    if "f4" in args.models:
        for name in dict.fromkeys(["full"] + args.interventions):
            result[name] = ("f4", INTERVENTIONS[name])
        for scale in args.alpha_scales:
            result[f"alpha_{scale:g}"] = ("f4", dict(low_scale=scale, high_scale=scale))
        for low, high in args.alpha_pairs:
            result[f"alpha_L{low:g}_H{high:g}"] = ("f4", dict(low_scale=low, high_scale=high))
    for repeat in range(1, args.native_repeats):
        for reference in ("baseline", "full"):
            if reference in result:
                result[f"{reference}_repeat{repeat}"] = result[reference]
    return result


def node_groups(x, c, r):
    masks = dict(all=torch.ones_like(x, dtype=torch.bool), selected_conflict=(x == 1) & (c > 0),
                 selected_clear=(x == 1) & (c == 0), addable=(x == 0) & (c == 0),
                 unique_blocker=(x == 0) & (c == 1), multi_blocker=(x == 0) & (c >= 2),
                 release_positive=r > 0)
    positive = r[r > 0]
    threshold = torch.quantile(positive, 0.75) if positive.numel() else None
    masks["release_high"] = r >= threshold if threshold is not None else torch.zeros_like(x, dtype=torch.bool)
    return masks


def same_state_rows(state, counts, release, scores, uniform, *, beta=1.02, binary=None, representations=None):
    """Particle-level conditional summaries; all comparisons use the same teacher state."""
    state, counts, release, uniform = [value.float().cpu() for value in (state, counts, release, uniform)]
    scores = {name: value.float().cpu() for name, value in scores.items()}
    reference = scores["full"]
    reference_probability = reference.sigmoid()
    delta_energy = (1 - 2 * state) * (beta * counts - 1)
    binary = binary.float().cpu() if binary is not None else None
    current_energy = (state * (beta * counts / 2 - 1)).sum(0)
    masks, rows = node_groups(state, counts, release), []
    for name, logit in scores.items():
        probability = logit.sigmoid()
        representation = (representations or {}).get(name)
        expected_joint, sampled_joint = None, None
        if binary is not None:
            expected_state = state + (1 - 2 * state) * probability
            sampled_state = torch.where(uniform < probability, 1 - state, state)
            expected_joint = (expected_state * (beta * (binary @ expected_state) / 2 - 1)).sum(0) - current_energy
            sampled_joint = (sampled_state * (beta * (binary @ sampled_state) / 2 - 1)).sum(0) - current_energy
        for particle in range(state.shape[1]):
            for category, mask in masks.items():
                chosen = mask[:, particle]
                size = int(chosen.sum())
                if not size:
                    continue
                p, ref = probability[chosen, particle], reference_probability[chosen, particle]
                dl = logit[chosen, particle] - reference[chosen, particle]
                dp = p - ref
                energy = delta_energy[chosen, particle]
                count = min(5, size)
                top = set(p.topk(count).indices.tolist())
                top_ref = set(ref.topk(count).indices.tolist())
                centered_score = logit[chosen, particle] - logit[chosen, particle].mean()
                centered_gain = -energy + energy.mean()
                denominator = centered_score.norm() * centered_gain.norm()
                changes = dl.abs()
                k = max(1, int(size * 0.1))
                rows.append(dict(
                    variant=name, particle=particle, category=category, nodes=size,
                    delta_probability_mean=dp.mean().item(), delta_probability_abs_mean=dp.abs().mean().item(),
                    delta_probability_p90=torch.quantile(dp.abs(), .9).item(),
                    delta_probability_p99=torch.quantile(dp.abs(), .99).item(),
                    probability_increase_fraction=(dp > 0).float().mean().item(),
                    expected_flips=p.sum().item(), expected_flips_reference=ref.sum().item(),
                    flips_target_error=(p.sum().item() - 5) if category == "all" else None,
                    flips_target_squared_error=(p.sum().item() - 5) ** 2 if category == "all" else None,
                    actual_flips=int((uniform[chosen, particle] < p).sum()),
                    sampled_action_disagreement=((uniform[chosen, particle] < p) != (uniform[chosen, particle] < ref)).float().mean().item(),
                    top5_jaccard=len(top & top_ref) / len(top | top_ref),
                    delta_logit_mean=dl.mean().item(), delta_logit_rms=dl.square().mean().sqrt().item(),
                    delta_logit_top10pct_mass=changes.topk(k).values.sum().div(changes.sum().clamp_min(1e-30)).item(),
                    probability_below_001=(p < .01).float().mean().item(),
                    probability_above_099=(p > .99).float().mean().item(),
                    single_flip_delta_energy_mean=energy.mean().item(),
                    expected_sum_single_flip_delta_energy=(p * energy).sum().item(),
                    expected_joint_delta_energy=expected_joint[particle].item() if category == "all" and expected_joint is not None else None,
                    sampled_joint_delta_energy=sampled_joint[particle].item() if category == "all" and sampled_joint is not None else None,
                    favorable_probability_mass=p[energy < 0].sum().div(p.sum().clamp_min(1e-30)).item(),
                    logit_immediate_gain_correlation=((centered_score * centered_gain).sum() / denominator).item() if denominator > 0 else None,
                    selected_conflict_edges=(state[:, particle] * counts[:, particle]).sum().item() / 2,
                    addable_nodes=int(((state[:, particle] == 0) & (counts[:, particle] == 0)).sum()),
                    unique_blocker_nodes=int(((state[:, particle] == 0) & (counts[:, particle] == 1)).sum()),
                    gate_low_mean=representation["gates"][chosen, particle, 0].mean().item() if representation else None,
                    gate_high_mean=representation["gates"][chosen, particle, 1].mean().item() if representation else None,
                    residual_low_relative=(representation["delta_low_norm"][chosen, particle].norm() /
                                           representation["raw_norm"][chosen, particle].norm().clamp_min(1e-30)).item() if representation else None,
                    residual_high_relative=(representation["delta_high_norm"][chosen, particle].norm() /
                                            representation["raw_norm"][chosen, particle].norm().clamp_min(1e-30)).item() if representation else None,
                ))
    return rows


def build_parser():
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--graphs", type=int, default=100)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_MODES), default=list(MODEL_MODES))
    parser.add_argument("--interventions", nargs="+", choices=tuple(INTERVENTIONS),
                        default=["full", "no_low", "no_high", "gate_half", "no_adapter"])
    parser.add_argument("--alpha-scales", nargs="*", type=float, default=[])
    parser.add_argument("--alpha-pairs", nargs=2, type=float, action="append", default=[])
    parser.add_argument("--mode", choices=("same_state", "replay", "both"), default="both")
    parser.add_argument("--phases", nargs="+", type=int, default=list(PHASES))
    parser.add_argument("--diagnostic-particles", type=int, default=4)
    parser.add_argument("--post", nargs="+", choices=("none", "pave"), default=["none", "pave"])
    parser.add_argument("--native-repeats", type=int, default=1,
                        help="Total baseline and Full replays under the same x0/U, including their first run")
    return parser


@torch.no_grad()
def run(args):
    if "f4" not in args.models and args.mode != "replay":
        raise ValueError("Same-state comparisons require the full F4 reference")
    if not 1 <= args.diagnostic_particles <= 20 or args.graphs <= 0 or args.native_repeats < 1:
        raise ValueError("Choose 1..20 diagnostic particles and a positive graph count")
    if set(args.phases) - set(range(101)):
        raise ValueError("Observation phases must lie in 0..100")
    torch.set_num_threads(1)
    solvers, paths = load_group(args, list(dict.fromkeys(["baseline"] + args.models)))
    teacher = solvers["baseline"]
    device = teacher.accelerator.device
    dataset = dataset_at(args.test_path, teacher.dtype, args.graphs)
    choices = variants(args)
    ranker = None
    pave_path = ROOT / f"outputs/v3/stats_h1_seed{args.seed}/checkpoints/best.pt"
    if args.mode != "same_state" and "pave" in args.post:
        checkpoint = torch.load(pave_path, map_location="cpu", weights_only=False)
        ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24)).eval().requires_grad_(False)
        ranker.load_state_dict(checkpoint["model_state"])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(version="v4_frequency_followup", mode="branch_replay", status="running", config=vars(args),
                    seed=args.seed, precision=args.precision, split=args.split, checkpoints=paths,
                    started_at=datetime.now(timezone.utc).isoformat(), variants=choices,
                    random_protocol="One x0 and 100 stacked per-step U tensors shared across variants; extra independent U for phase 100 only",
                    precision_protocol="Compare within precision; fp32 and fp16 use their original dtype-specific random streams",
                    native_repeats=args.native_repeats,
                    native_numerics="Shared x0/U does not remove CUDA sparse-reduction nondeterminism; native repeat controls quantify its trajectory and Size effects",
                    archive="Per-particle minimum energy followed by one final A.3 decode; best legal set over K",
                    same_state_source="Frozen original RLNN baseline", same_state_reference="Full F4 at the same teacher state",
                    immediate_energy="expected_joint_delta_energy is exact under independent Bernoulli flips; expected_sum_single_flip_delta_energy omits simultaneous-flip interactions",
                    timing="Replay and same-state timings are diagnostic costs, not official E2E timing",
                    pave_checkpoint=str(pave_path) if ranker is not None else None)
    write_run(args.result_dir, metadata)
    started = perf_counter()
    final_rows, state_rows, curve_rows, repeat_rows = [], [], [], []
    try:
        for index, item in enumerate(dataset):
            graph_id = Path(dataset.files[index]).stem
            cpu_graph = SearchGraph(item.num_nodes, item.edge_index, seed=args.search_seed + index)
            graph = item.to(device)
            adj = adjacency(graph)
            topologies = {name: solver.policy_topology(graph, adj) for name, solver in solvers.items()}
            ft = topologies.get("f4") or v4_topology(graph.edge_index, adj, graph.num_nodes)
            initial, uniforms, terminal_probe = shared_noise(graph.num_nodes, args.num_k, args.num_t,
                                                            device, teacher.dtype, args.search_seed + index)
            results = {}
            sync(device)
            for variant, (name, intervention) in {"baseline": ("baseline", {}), **choices}.items():
                if args.mode == "same_state" and variant != "baseline":
                    continue
                solver = solvers[name]
                forward = lambda x, s=solver, t=topologies[name], i=intervention: policy_logits(s, graph, adj, t, x, i)
                tick = perf_counter()
                result = replay(solver.problem, adj, graph.b, initial, uniforms, forward,
                                phases=args.phases, capture=variant in ("baseline", "full") or "_repeat" in variant)
                sync(device)
                elapsed = perf_counter() - tick
                selected = result["selected"]
                if not cpu_graph.feasible(selected) or (graph.num_nodes and not selected):
                    raise ValueError(f"Invalid replay incumbent: {graph_id}/{variant}")
                results[variant] = result
                if args.mode == "same_state" or variant not in choices:
                    continue
                final, post_seconds = selected, 0.0
                if ranker is not None:
                    tick = perf_counter()
                    final = platform_search(cpu_graph, selected, choose=ranker.choose, horizon=1,
                                            initial_rounds=3, outer_rounds=8, candidate_cap=32,
                                            improving_cap=48, repair_rounds=4).best
                    post_seconds = perf_counter() - tick
                    if not cpu_graph.feasible(final):
                        raise ValueError(f"Invalid PAVE incumbent: {graph_id}/{variant}")
                common = dict(graph_id=graph_id, graph_index=index, seed=args.seed, precision=args.precision,
                              variant=variant, model=name, search_seed=args.search_seed + index)
                final_rows.append(dict(**common, base_size=selected.bit_count(), final_size=final.bit_count(),
                                       pave_gain=final.bit_count() - selected.bit_count(),
                                       replay_diagnostic_time_s=elapsed, pave_time_s=post_seconds,
                                       base_nodes=" ".join(map(str, nodes(selected))),
                                       final_nodes=" ".join(map(str, nodes(final)))))
                result["pave_selected"] = final
                curve_rows.extend({**common, **row} for row in result["curves"])
            if args.mode != "same_state":
                for variant, result in results.items():
                    if "_repeat" not in variant:
                        continue
                    reference_name = variant.split("_repeat", 1)[0]
                    reference = results[reference_name]
                    changed = {step: int((result["states"][step] != reference["states"][step]).sum())
                               for step in result["states"]}
                    repeat_rows.append(dict(
                        graph_id=graph_id, graph_index=index, seed=args.seed, precision=args.precision,
                        variant=variant, reference_variant=reference_name,
                        base_size_delta=result["selected"].bit_count() - reference["selected"].bit_count(),
                        final_size_delta=result["pave_selected"].bit_count() - reference["pave_selected"].bit_count(),
                        selected_nodes_changed=result["selected"] != reference["selected"],
                        final_state_changed_entries=int((result["final_state"] != reference["final_state"]).sum()),
                        best_archive_changed_entries=int((result["best_states"] != reference["best_states"]).sum()),
                        changed_observation_phases=sum(count > 0 for count in changed.values()),
                        changed_observed_state_entries=sum(changed.values()),
                        first_changed_observation_phase=next((step for step, count in changed.items() if count), None)))
            if args.mode != "replay":
                for step, state in results["baseline"]["states"].items():
                    x = state[:, :args.diagnostic_particles]
                    counts = ft.binary @ x.float()
                    release = x.float() * (ft.binary @ ((x == 0) & (counts == 1)).float())
                    noise = (uniforms[step] if step < args.num_t else terminal_probe)[:, :args.diagnostic_particles]
                    scores = {variant: policy_logits(solvers[name], graph, adj, topologies[name], x, intervention)
                              for variant, (name, intervention) in choices.items()}
                    representations = {}
                    for variant, (name, intervention) in choices.items():
                        solver = solvers[name]
                        model = solver.accelerator.unwrap_model(solver.model)
                        if not hasattr(model, "adapter"):
                            continue
                        with solver.accelerator.autocast():
                            details = intervention_details(model.adapter, x, adj, topologies[name], **intervention)
                            raw = model.initial_project(x.unsqueeze(-1))
                        representations[variant] = dict(
                            gates=details["gates"].float().cpu(), raw_norm=raw.float().norm(dim=-1).cpu(),
                            delta_low_norm=details["delta_low"].float().norm(dim=-1).cpu(),
                            delta_high_norm=details["delta_high"].float().norm(dim=-1).cpu())
                    common = dict(graph_id=graph_id, graph_index=index, seed=args.seed, precision=args.precision, step=step)
                    state_rows.extend({**common, **row} for row in same_state_rows(x, counts, release, scores, noise,
                                                                                binary=ft.binary, representations=representations))
            if (index + 1) % 5 == 0 or index + 1 == len(dataset):
                print(f"{index + 1}/{len(dataset)} paired graphs completed", flush=True)
        for name, rows in (("replay_per_graph.csv", final_rows), ("same_state.csv", state_rows),
                           ("replay_curves.csv", curve_rows), ("native_repeat_noise.csv", repeat_rows)):
            csv_file(args.result_dir / name, rows)
        metadata.update(status="completed", samples=len(dataset), graph_ids=[Path(p).stem for p in dataset.files],
                        finished_at=datetime.now(timezone.utc).isoformat(), wall_time_s=perf_counter() - started)
    except Exception as error:
        metadata.update(status="failed", error=str(error), wall_time_s=perf_counter() - started)
        write_run(args.result_dir, metadata)
        raise
    write_run(args.result_dir, metadata)
    return final_rows


if __name__ == "__main__":
    run(build_parser().parse_args())
