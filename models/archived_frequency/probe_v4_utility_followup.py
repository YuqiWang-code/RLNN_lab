"""Shared-label flip diagnostics with nested rollout repeats and stratified nodes."""
import argparse
import csv
from datetime import datetime, timezone
import json
from math import sqrt
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v2 import TRAIN, VALID, dataset_at
from mis_v1 import adjacency
from archived_frequency.mis_v4_frequency import state_bands, v4_topology
from models import edge2adj
from archived_frequency.probe_v4_utility import checkpoint_file, load_model, logits, rank_metrics, write_csv
from problems.mis import Problem


ROOT = Path(__file__).resolve().parents[2]
LABEL_MODES = {"B": "none", "BFT": "none", "MLP": "v4a_mlp", "MSG": "v4a_message", "F4": "v4a"}
CATEGORIES = ("selected_conflicted", "selected_clear", "unselected_free", "unselected_one", "unselected_many")


def model_paths(seed):
    return {
        "B": ROOT / f"outputs/v2/none_seed{seed}/checkpoints/best.pt",
        "BFT": ROOT / f"outputs/v4_frequency/none_seed{seed}/checkpoints/best.pt",
        "MLP": ROOT / f"outputs/v4_frequency/v4a_mlp_seed{seed}/checkpoints/best.pt",
        "MSG": ROOT / f"outputs/v4_frequency/v4a_message_seed{seed}/checkpoints/best.pt",
        "F4": ROOT / f"outputs/v4_frequency/v4a_seed{seed}/checkpoints/best.pt",
    }


def node_masks(state, counts, release):
    """Five disjoint state categories plus an overlapping upper-quartile release group."""
    x, c, r = state.cpu(), counts.cpu(), release.cpu()
    masks = dict(zip(CATEGORIES, (
        (x == 1) & (c > 0), (x == 1) & (c == 0), (x == 0) & (c == 0),
        (x == 0) & (c == 1), (x == 0) & (c >= 2),
    )))
    positive = r[r > 0]
    threshold = float(torch.quantile(positive, 0.75)) if positive.numel() else None
    masks["release_high"] = (r >= threshold) & (r > 0) if threshold is not None else torch.zeros_like(x, dtype=torch.bool)
    return masks, threshold


def sample_candidates(masks, count, sampling, seed):
    """Sample unique nodes, with explicit quotas and a uniform fill for sparse strata."""
    nodes = len(next(iter(masks.values())))
    count = min(count, nodes)
    generator = torch.Generator().manual_seed(seed)
    if sampling == "uniform":
        return torch.randperm(nodes, generator=generator)[:count].sort().values
    selected = torch.zeros(nodes, dtype=torch.bool)
    order = ("release_high",) + CATEGORIES
    for index, name in enumerate(order):
        quota = count // len(order) + int(index < count % len(order))
        available = (masks[name] & ~selected).nonzero().flatten()
        choices = available[torch.randperm(len(available), generator=generator)[:quota]]
        selected[choices] = True
    remaining = (~selected).nonzero().flatten()
    needed = count - int(selected.sum())
    selected[remaining[torch.randperm(len(remaining), generator=generator)[:needed]]] = True
    return selected.nonzero().flatten()


def continuation_noise(nodes, horizon, replicates, seed, device):
    """Generate per-replicate streams with the original one-vector-per-step RNG calls."""
    streams = []
    for repeat in range(replicates):
        generator = torch.Generator(device=device).manual_seed(seed + repeat)
        streams.append(torch.stack([torch.rand(nodes, device=device, generator=generator) for _ in range(horizon)]))
    return torch.stack(streams)


@torch.no_grad()
def paired_utilities(teacher, state, candidates, graph, adj, normalized, problem,
                     horizon, seed, replicates=8, chunk_size=256):
    """Batch independent action/repeat columns, sharing noise with no-flip in each repeat."""
    choices = candidates.numel() + 1
    noise = continuation_noise(len(state), horizon, replicates, seed, state.device)
    sizes = torch.empty(replicates * choices, device=state.device)
    for first in range(0, len(sizes), chunk_size):
        columns = torch.arange(first, min(first + chunk_size, len(sizes)), device=state.device)
        repeat_ids, action_ids = columns // choices, columns % choices
        x = state[:, None].expand(-1, len(columns)).clone()
        changed = (action_ids > 0).nonzero().flatten()
        nodes = candidates[action_ids[changed] - 1]
        x[nodes, changed] = 1 - x[nodes, changed]
        best = x.clone()
        best_energy, _ = problem.energy_func(adj, graph.b, x)
        for step in range(horizon):
            probability = teacher(x.unsqueeze(-1), graph.edge_index, normalized).squeeze(-1).sigmoid()
            x = torch.where(noise[repeat_ids, step].T < probability, 1 - x, x)
            energy, _ = problem.energy_func(adj, graph.b, x)
            improves = energy < best_energy
            best = torch.where(improves.unsqueeze(0), x, best)
            best_energy = torch.minimum(best_energy, energy)
        solutions = problem.decode_gradient_flip(best, adj, graph.b, return_solutions=True)
        if ((adj @ solutions) * solutions).any():
            raise ValueError("A.3 produced a conflicting independent set")
        sizes[columns] = solutions.sum(0)
    sizes = sizes.reshape(replicates, choices)
    return sizes, sizes[:, 1:] - sizes[:, :1]


def prefix_counts(replicates):
    return [value for value in (2, 4, 8, 16) if value <= replicates]


def average_ranks(values):
    differences = values[:, None] - values[None, :]
    return (differences > 0).sum(1).float() + 0.5 * (differences == 0).sum(1).float()


def label_reliability(gains):
    """Compare independent first/second halves of the same nested repeat prefix."""
    repeats = len(gains)
    first, second = gains[:repeats // 2].mean(0), gains[repeats // 2:].mean(0)
    pair_indices = torch.triu_indices(gains.shape[1], gains.shape[1], offset=1)
    a = first[pair_indices[0]] - first[pair_indices[1]]
    b = second[pair_indices[0]] - second[pair_indices[1]]
    both = (a != 0) & (b != 0)
    direction_matches = int(((a * b > 0) & both).sum())
    u, v = average_ranks(first), average_ranks(second)
    u, v = u - u.mean(), v - v.mean()
    denominator = u.norm() * v.norm()
    first_best, second_best = first == first.max(), second == second.max()
    means = gains.mean(0)
    differences = means[pair_indices[0]] - means[pair_indices[1]]
    return dict(
        pairs=pair_indices.shape[1], non_tied_pairs=int((differences != 0).sum()),
        informative=int((differences != 0).any()),
        nonzero_mean_fraction=float((means != 0).float().mean()),
        positive_mean_fraction=float((means > 0).float().mean()),
        zero_replicate_fraction=float((gains == 0).float().mean()),
        mean_gain_sem=float((gains.std(0, unbiased=True) / sqrt(repeats)).mean()),
        half_spearman=float((u @ v) / denominator) if denominator > 0 else None,
        half_top1_same=int(first.argmax() == second.argmax()),
        half_top1_set_jaccard=float((first_best & second_best).sum() / (first_best | second_best).sum()),
        half_action_sign_agreement=float((first.sign() == second.sign()).float().mean()),
        half_joint_non_tied_pairs=int(both.sum()), half_direction_match_pairs=direction_matches,
        half_pair_direction_agreement=direction_matches / int(both.sum()) if both.any() else None,
        mean_gain=float(means.mean()), oracle_sample_gain=float(means.max()),
    )


@torch.no_grad()
def score_actions(models, x, graph, adj, topology, normalized, details, seed):
    scores = {name: logits(model, LABEL_MODES[name], x, graph, adj, topology, normalized)[:, 0].cpu()
              for name, model in models.items()}
    adapter = models["F4"].adapter
    saved = adapter.low_override, adapter.high_override
    try:
        adapter.low_override, adapter.high_override = False, None
        scores["F4_no_low"] = logits(models["F4"], "v4a", x, graph, adj, topology, normalized)[:, 0].cpu()
        adapter.low_override, adapter.high_override = None, False
        scores["F4_no_high"] = logits(models["F4"], "v4a", x, graph, adj, topology, normalized)[:, 0].cpu()
    finally:
        adapter.low_override, adapter.high_override = saved
    scores.update(
        Energy=-details["delta_energy"][:, 0].cpu(),
        Random=torch.rand(graph.num_nodes, generator=torch.Generator().manual_seed(seed)),
        DegreeLow=-topology.degree[:, 0].cpu(),
        SelectedNeighbors=details["c"][:, 0].cpu(),
        Release=details["r"][:, 0].cpu(),
    )
    return scores


def graph_average(rows, key):
    graph_values = {}
    for row in rows:
        if row[key] is not None:
            graph_values.setdefault(row["graph_id"], []).append(row[key])
    return mean(mean(values) for values in graph_values.values()) if graph_values else None


def summarize_rankings(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["R"], row["candidate_group"], row["model"]), []).append(row)
    summaries = {}
    for (repeats, group, model), entries in groups.items():
        informative = [row for row in entries if row["non_tied_pairs"]]
        pairs = sum(row["non_tied_pairs"] for row in entries)
        summary = dict(
            graphs=len({row["graph_id"] for row in entries}), states=len(entries),
            informative_states=len(informative), non_tied_pairs=pairs,
            graph_mean_pair_accuracy=graph_average(entries, "pair_accuracy"),
            state_mean_pair_accuracy=mean(row["pair_accuracy"] for row in informative) if informative else None,
            pooled_pair_accuracy=sum(row["correct_pair_credit"] for row in entries) / pairs if pairs else None,
            mean_top1_gain=graph_average(entries, "top1_gain"), mean_top1_regret=graph_average(entries, "top1_regret"),
            mean_uniform_gain=graph_average(entries, "uniform_gain"),
            mean_oracle_sample_gain=graph_average(entries, "oracle_sample_gain"),
            top1_positive_fraction=graph_average(entries, "top1_positive"),
        )
        summaries.setdefault(f"R{repeats}", {}).setdefault(group, {})[model] = summary
    return summaries


def write_records(path, rows):
    """Keep JSON-valued state metadata within ordinary CSV cells."""
    write_csv(path, [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                      for key, value in row.items()} for row in rows])


@torch.no_grad()
def probe(args):
    if args.graphs <= 0 or args.nodes < 2 or args.chunk_size < 1:
        raise ValueError("Use positive graph/chunk counts and at least two candidate nodes")
    if any(step < 0 or step > 100 for step in args.phases) or not args.phases:
        raise ValueError("Observation phases must be within 0..100")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    paths = model_paths(args.seed)
    for name, mode, path in args.model:
        if name not in LABEL_MODES or LABEL_MODES[name] != mode:
            raise ValueError(f"Use B/BFT/MLP/MSG/F4 with their defined mode: {name}, {mode}")
        paths[name] = checkpoint_file(path)
    models = {name: load_model(LABEL_MODES[name], path, device) for name, path in paths.items()}
    teacher = models["B"]
    trajectory_model = models[args.trajectory_model]
    problem = Problem(1.02)
    data_path = args.data_path or (TRAIN if args.split == "train" else VALID)
    dataset = dataset_at(data_path, torch.float32, args.graphs)
    if len(dataset) != args.graphs:
        raise ValueError(f"Requested {args.graphs} graphs, found {len(dataset)}")
    args.result_dir.mkdir(parents=True, exist_ok=False)
    phases = sorted(set(args.phases))
    metadata = dict(
        version="v4_frequency_followup", mode="utility_probe", status="running", split=args.split,
        seed=args.seed, config=vars(args), checkpoints={name: str(path) for name, path in paths.items()},
        graphs=len(dataset), graph_ids=[Path(file).stem for file in dataset.files], graph_files=dataset.files,
        started_at=datetime.now(timezone.utc).isoformat(), precision="float32", postprocess="none",
        trajectory_model=args.trajectory_model, trajectory_particles=1, continuation_model="B",
        continuation_policy="frozen baseline; labels shared by all scoring models", continuation_steps=args.horizon,
        replicates=args.replicates, replicate_prefixes=prefix_counts(args.replicates), phases=phases,
        score_definition="Raw flip logits at the same state; highest score selected; candidate node IDs break score ties",
        rule_scores={"Energy": "negative exact single-flip energy change", "Random": "fixed uniform random score",
                     "DegreeLow": "negative degree", "SelectedNeighbors": "selected-neighbor count c", "Release": "release count r"},
        sampling=args.sampling, release_high_definition="r>0 and r>=75th percentile among positive r; ties included",
        sampling_definition="Uniform or equal quotas across release_high then A-E; unique nodes, uniform fill, sorted IDs",
        rng_definition="state_code=(train_seed*1000000+graph_index)*128+step; continuation=1000000000+32*state_code+repeat; candidate=2000000000+state_code; random_score=3000000000+state_code",
        aggregation="state metrics first, graph means next; non-tied-state and pooled-pair accuracy also retained",
        protocol="Single-node forced flip versus no flip, common per-repeat noise across candidates, per-column energy archive and A.3",
        timing="Diagnostic wall time; excluded from formal solver-time comparisons",
    )

    def save():
        (args.result_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    started = perf_counter()
    save()
    state_rows, repeat_rows, aggregate_rows, ranking_rows, reliability_rows = [], [], [], [], []
    try:
        for index, graph in enumerate(dataset):
            graph_id = Path(dataset.files[index]).stem
            graph = graph.to(device)
            adj = adjacency(graph)
            normalized = edge2adj(graph.edge_index, graph.num_nodes)
            topology = v4_topology(graph.edge_index, adj, graph.num_nodes)
            trajectory_rng = torch.Generator(device=device).manual_seed(args.search_seed + index)
            x = torch.randint(0, 2, (graph.num_nodes, 1), device=device, generator=trajectory_rng).float()
            for step in range(max(phases) + 1):
                if step > 0:
                    probability = logits(trajectory_model, LABEL_MODES[args.trajectory_model], x, graph, adj, topology, normalized).sigmoid()
                    noise = torch.rand(x.shape, device=device, generator=trajectory_rng)
                    x = torch.where(noise < probability, 1 - x, x)
                if step not in phases:
                    continue
                code = (args.seed * 1000000 + index) * 128 + step
                candidate_seed, continuation_seed, score_seed = 2000000000 + code, 1000000000 + 32 * code, 3000000000 + code
                details = state_bands(x, topology)
                masks, release_threshold = node_masks(x[:, 0], details["c"][:, 0], details["r"][:, 0])
                candidates_cpu = sample_candidates(masks, args.nodes, args.sampling, candidate_seed)
                candidates = candidates_cpu.to(device)
                sizes, gains = paired_utilities(teacher, x[:, 0], candidates, graph, adj, normalized, problem,
                    args.horizon, continuation_seed, args.replicates, args.chunk_size)
                sizes, gains = sizes.cpu(), gains.cpu()
                scores = {name: values[candidates_cpu] for name, values in score_actions(
                    models, x, graph, adj, topology, normalized, details, score_seed).items()}
                common = dict(graph_id=graph_id, graph_index=index, seed=args.seed,
                              state_id=f"{args.trajectory_model}:{graph_id}:t{step}", step=step,
                              trajectory_model=args.trajectory_model, continuation_model="B", candidate_sampling=args.sampling)
                groups = {"all": torch.arange(len(candidates_cpu))}
                groups.update({name: mask[candidates_cpu].nonzero().flatten() for name, mask in masks.items()})
                state_rows.append(dict(**common, nodes=graph.num_nodes, candidates=len(candidates_cpu),
                    selected_nodes=x[:, 0].nonzero().flatten().cpu().tolist(), candidate_nodes=candidates_cpu.tolist(),
                    release_high_threshold=release_threshold, trajectory_seed=args.search_seed + index,
                    candidate_seed=candidate_seed, continuation_seed_base=continuation_seed,
                    group_available={name: int(mask.sum()) for name, mask in masks.items()},
                    group_selected={name: len(groups[name]) for name in masks},
                    group_sample_fraction={name: len(groups[name]) / int(mask.sum()) if mask.any() else None for name, mask in masks.items()}))
                for column, node in enumerate(candidates_cpu.tolist()):
                    category = next(name for name in CATEGORIES if masks[name][node])
                    action = dict(**common, node=node, category=category, release_high=int(masks["release_high"][node]),
                                  selected=int(x[node, 0]), selected_neighbors=float(details["c"][node, 0]),
                                  release_count=float(details["r"][node, 0]))
                    for repeat in range(args.replicates):
                        repeat_rows.append(dict(**action, replicate=repeat, continuation_seed=continuation_seed + repeat,
                            noflip_size=int(sizes[repeat, 0]), flip_size=int(sizes[repeat, column + 1]), gain=int(gains[repeat, column])))
                    for repeats in prefix_counts(args.replicates):
                        samples = gains[:repeats, column]
                        row = dict(**action, R=repeats, gain_mean=float(samples.mean()),
                            gain_std=float(samples.std(unbiased=True)), gain_sem=float(samples.std(unbiased=True)) / sqrt(repeats),
                            positive_fraction=float((samples > 0).float().mean()), negative_fraction=float((samples < 0).float().mean()),
                            zero_fraction=float((samples == 0).float().mean()))
                        row.update({f"score_{name}": float(values[column]) for name, values in scores.items()})
                        aggregate_rows.append(row)
                for repeats in prefix_counts(args.replicates):
                    utilities = gains[:repeats].mean(0)
                    for group_name, columns in groups.items():
                        if not len(columns):
                            continue
                        reliability_rows.append(dict(**common, R=repeats, candidate_group=group_name,
                            candidates=len(columns), **label_reliability(gains[:repeats, columns])))
                        for name, values in scores.items():
                            metrics = rank_metrics(values[columns], utilities[columns])
                            chosen_column = int(columns[metrics.pop("chosen_index")])
                            metrics["oracle_sample_gain"] = metrics.pop("oracle_flip_gain")
                            ranking_rows.append(dict(**common, R=repeats, candidate_group=group_name,
                                candidates=len(columns), model=name, chosen_node=int(candidates_cpu[chosen_column]),
                                top1_positive=int(metrics["top1_gain"] > 0), **metrics))
            print(f"{index + 1}/{len(dataset)} {args.split} graphs; {len(state_rows)} states", flush=True)
        for filename, rows in (("state.csv", state_rows), ("action_replicate.csv", repeat_rows),
                               ("action_aggregate.csv", aggregate_rows), ("ranking.csv", ranking_rows),
                               ("reliability.csv", reliability_rows)):
            write_records(args.result_dir / filename, rows)
        metadata.update(status="completed", states=len(state_rows),
                        actions=sum(row["candidates"] for row in state_rows), action_replicates=len(repeat_rows),
                        summaries=summarize_rankings(ranking_rows))
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), wall_time_s=perf_counter() - started)
        save()
    return metadata


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--model", nargs=3, action="append", default=[], metavar=("NAME", "MODE", "CHECKPOINT"))
    parser.add_argument("--graphs", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--replicates", type=int, choices=(2, 4, 8, 16), default=8)
    parser.add_argument("--sampling", choices=("uniform", "stratified"), default="uniform")
    parser.add_argument("--phases", type=int, nargs="+", default=[25, 50, 100])
    parser.add_argument("--horizon", type=int, choices=(10, 20, 40), default=20)
    parser.add_argument("--trajectory-model", choices=tuple(LABEL_MODES), default="B",
                        help="Policy generating observed states; all action labels still use frozen B continuation")
    parser.add_argument("--search-seed", type=int, default=1017)
    parser.add_argument("--chunk-size", type=int, default=256, help="Maximum action/repeat columns per teacher forward")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    probe(build_parser().parse_args())
