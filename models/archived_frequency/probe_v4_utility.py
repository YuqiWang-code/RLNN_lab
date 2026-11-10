"""Validation-only paired flip interventions under a frozen RLNN continuation."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v2 import VALID, dataset_at
from mis_v1 import adjacency
from archived_frequency.mis_v4_frequency import V4FrequencyGCN, V4_MODES, v4_topology
from models import GCN, edge2adj
from problems.mis import Problem


ROOT = Path(__file__).resolve().parents[2]


def checkpoint_file(path):
    path = Path(path)
    return path / "best.pt" if path.is_dir() else path


def load_model(mode, path, device):
    model = GCN(1, 128, 5, 1) if mode == "none" else V4FrequencyGCN(128, 5, mode=mode)
    model.load_state_dict(torch.load(checkpoint_file(path), map_location="cpu", weights_only=True))
    return model.to(device).eval().requires_grad_(False)


def logits(model, mode, x, graph, adj, topology, normalized):
    if mode == "none":
        return model(x.unsqueeze(-1), graph.edge_index, normalized).squeeze(-1)
    return model(x, graph.edge_index, adj, topology)


@torch.no_grad()
def paired_utilities(teacher, state, candidates, graph, adj, normalized, problem,
                     horizon, seed, replicates=2):
    """Share continuation noise across all interventions within each replicate."""
    choices = candidates.numel() + 1
    x = state[:, None, None].expand(-1, replicates, choices).clone()
    candidate_columns = torch.arange(1, choices, device=state.device)
    for replicate in range(replicates):
        x[candidates, replicate, candidate_columns] = 1 - x[candidates, replicate, candidate_columns]
    x = x.flatten(1)
    best = x.clone()
    best_energy, _ = problem.energy_func(adj, graph.b, x)
    generators = [torch.Generator(device=state.device).manual_seed(seed + replicate)
                  for replicate in range(replicates)]
    for _ in range(horizon):
        probability = teacher(x.unsqueeze(-1), graph.edge_index, normalized).squeeze(-1).sigmoid()
        shared_noise = torch.stack([
            torch.rand(state.shape[0], generator=generator, device=state.device)
            for generator in generators
        ], dim=1)
        noise = shared_noise.unsqueeze(-1).expand(-1, -1, choices).flatten(1)
        x = torch.where(noise < probability, 1 - x, x)
        energy, _ = problem.energy_func(adj, graph.b, x)
        improves = energy < best_energy
        best = torch.where(improves.unsqueeze(0), x, best)
        best_energy = torch.minimum(best_energy, energy)
    solutions = problem.decode_gradient_flip(best, adj, graph.b, return_solutions=True)
    if ((adj @ solutions) * solutions).any():
        raise ValueError("A.3 produced a conflicting independent set")
    sizes = solutions.sum(0).reshape(replicates, choices)
    gains = sizes[:, 1:] - sizes[:, :1]
    return sizes, gains


def rank_metrics(scores, utilities):
    pairs = torch.triu_indices(utilities.numel(), utilities.numel(), offset=1)
    differences = utilities[pairs[0]] - utilities[pairs[1]]
    non_ties = differences != 0
    score_differences = scores[pairs[0]] - scores[pairs[1]]
    products = score_differences[non_ties] * differences[non_ties]
    correct = (products > 0).float() + 0.5 * (products == 0).float()
    chosen = int(scores.argmax())
    return dict(
        pairs=pairs.shape[1], non_tied_pairs=int(non_ties.sum()),
        correct_pair_credit=float(correct.sum()),
        pair_accuracy=float(correct.mean()) if correct.numel() else None,
        score_tied_pairs=int((score_differences[non_ties] == 0).sum()),
        chosen_index=chosen, top1_gain=float(utilities[chosen]),
        uniform_gain=float(utilities.mean()), oracle_flip_gain=float(utilities.max()),
        top1_regret=float(utilities.max() - utilities[chosen]),
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def probe(args):
    if args.graphs <= 0 or args.nodes < 2:
        raise ValueError("The probe requires positive graph count and at least two sampled nodes")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    models = {"baseline": ("none", load_model("none", args.teacher_checkpoint, device))}
    paths = {"baseline": str(checkpoint_file(args.teacher_checkpoint))}
    for name, mode, path in args.model:
        if name in models or mode not in ("none",) + V4_MODES:
            raise ValueError(f"Duplicate name or unsupported mode: {name}, {mode}")
        models[name] = mode, load_model(mode, path, device)
        paths[name] = str(checkpoint_file(path))
    teacher = models["baseline"][1]
    problem = Problem(1.02)
    dataset = dataset_at(args.validation_path, torch.float32, args.graphs)
    if not len(dataset):
        raise ValueError("The validation dataset is empty")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        version="v4_frequency", mode="utility_probe", status="running", split="validation",
        seed=args.seed, config=vars(args), checkpoints=paths,
        started_at=datetime.now(timezone.utc).isoformat(),
        protocol="Current single-trajectory states; paired node flips with common continuation noise; per-column energy archive and A.3",
        continuation_policy="frozen baseline", continuation_steps=args.horizon, replicates=2,
        times=[25, 50, 100], precision="float32", model_selection="diagnostic only",
    )

    def save():
        (args.result_dir / "run.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    save()
    started = perf_counter()
    action_rows, state_rows, ranking_rows = [], [], []
    try:
        for index, graph in enumerate(dataset):
            graph_id = Path(dataset.files[index]).stem
            graph = graph.to(device)
            adj = adjacency(graph)
            normalized = edge2adj(graph.edge_index, graph.num_nodes)
            topology = v4_topology(graph.edge_index, adj, graph.num_nodes)
            trajectory_rng = torch.Generator(device=device).manual_seed(args.seed + index)
            x = torch.randint(0, 2, (graph.num_nodes, 1), device=device,
                              generator=trajectory_rng).float()
            for step in range(1, 101):
                probability = teacher(x.unsqueeze(-1), graph.edge_index, normalized).squeeze(-1).sigmoid()
                noise = torch.rand(x.shape, device=device, generator=trajectory_rng)
                x = torch.where(noise < probability, 1 - x, x)
                if step not in (25, 50, 100):
                    continue
                state_id = f"{graph_id}:t{step}"
                node_seed = args.seed + 10000 * (index + 1) + step
                candidate_rng = torch.Generator().manual_seed(node_seed)
                candidates = torch.randperm(graph.num_nodes, generator=candidate_rng)[:args.nodes].to(device)
                sizes, gains = paired_utilities(
                    teacher, x[:, 0], candidates, graph, adj, normalized, problem,
                    args.horizon, seed=node_seed + 1000000,
                )
                sizes, gains = sizes.cpu(), gains.cpu()
                utilities = gains.mean(0)
                scores = {name: logits(model, mode, x, graph, adj, topology, normalized)[candidates, 0].cpu()
                          for name, (mode, model) in models.items()}
                common = dict(graph_id=graph_id, graph_index=index, state_id=state_id, step=step)
                neutral_metrics = rank_metrics(scores["baseline"], utilities)
                state_rows.append(dict(
                    **common, nodes=graph.num_nodes, candidates=len(candidates),
                    pairs=neutral_metrics["pairs"], non_tied_pairs=neutral_metrics["non_tied_pairs"],
                    informative=int(neutral_metrics["non_tied_pairs"] > 0),
                    nonzero_gain_candidates=int((utilities != 0).sum()),
                    positive_gain_candidates=int((utilities > 0).sum()),
                    mean_gain=float(utilities.mean()), oracle_flip_gain=float(utilities.max()),
                    noflip_size_r0=float(sizes[0, 0]), noflip_size_r1=float(sizes[1, 0]),
                    replicate_sign_agreement=float((gains[0].sign() == gains[1].sign()).float().mean()),
                ))
                for column, node in enumerate(candidates.tolist()):
                    row = dict(
                        **common, node=node, selected=int(x[node, 0]),
                        flip_size_r0=float(sizes[0, column + 1]), flip_size_r1=float(sizes[1, column + 1]),
                        noflip_size_r0=float(sizes[0, 0]), noflip_size_r1=float(sizes[1, 0]),
                        gain_r0=float(gains[0, column]), gain_r1=float(gains[1, column]),
                        mean_gain=float(utilities[column]),
                    )
                    row.update({f"logit_{name}": float(values[column]) for name, values in scores.items()})
                    action_rows.append(row)
                for name, values in scores.items():
                    metrics = rank_metrics(values, utilities)
                    metrics["chosen_node"] = int(candidates[metrics.pop("chosen_index")])
                    ranking_rows.append(dict(**common, model=name, **metrics))
            print(f"{index + 1}/{len(dataset)} validation graphs; {len(state_rows)} states", flush=True)
        write_csv(args.result_dir / "actions.csv", action_rows)
        write_csv(args.result_dir / "states.csv", state_rows)
        write_csv(args.result_dir / "ranking.csv", ranking_rows)
        summaries = {}
        for name in models:
            rows = [row for row in ranking_rows if row["model"] == name]
            informative = [row for row in rows if row["non_tied_pairs"]]
            non_tied = sum(row["non_tied_pairs"] for row in rows)
            summaries[name] = dict(
                states=len(rows), informative_states=len(informative),
                state_mean_pair_accuracy=mean(row["pair_accuracy"] for row in informative) if informative else None,
                pooled_pair_accuracy=sum(row["correct_pair_credit"] for row in rows) / non_tied if non_tied else None,
                mean_top1_gain=mean(row["top1_gain"] for row in rows),
                mean_uniform_gain=mean(row["uniform_gain"] for row in rows),
                mean_oracle_flip_gain=mean(row["oracle_flip_gain"] for row in rows),
                mean_top1_regret=mean(row["top1_regret"] for row in rows),
            )
        total_pairs = sum(row["pairs"] for row in state_rows)
        metadata.update(
            status="completed", graphs=len(dataset), states=len(state_rows), actions=len(action_rows),
            informative_states=sum(row["informative"] for row in state_rows),
            informative_state_fraction=mean(row["informative"] for row in state_rows),
            informative_graphs=len({row["graph_id"] for row in state_rows if row["informative"]}),
            total_pairs=total_pairs, non_tied_pairs=sum(row["non_tied_pairs"] for row in state_rows),
            non_tied_pair_fraction=sum(row["non_tied_pairs"] for row in state_rows) / total_pairs if total_pairs else None,
            model_metrics=summaries,
        )
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), wall_time_s=perf_counter() - started)
        save()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path,
                        default=ROOT / "outputs/v2/none_seed0/checkpoints/best.pt")
    parser.add_argument("--model", nargs=3, action="append", default=[], metavar=("NAME", "MODE", "CHECKPOINT"))
    parser.add_argument("--validation-path", type=Path, default=VALID)
    parser.add_argument("--result-dir", type=Path, default=ROOT / "outputs/v4_frequency/utility_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graphs", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--horizon", type=int, choices=(10, 20), default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    probe(args)


if __name__ == "__main__":
    main()
