"""P0-B / P0-C: does an h-step macro transition carry information a per-node
independent model cannot reproduce?

Nothing is trained here.  Both diagnostics use the *frozen* RLNN policy and the
frozen PAVE-Stats H1 head, so every number is a property of the existing search.

P0-B (correlation)
    For a fixed (G, x) we draw R continuations of h steps from the frozen policy.
    The ``joint`` arm keeps them as they are.  The ``marginal`` arm shuffles, for
    every node independently, the R observed flip decisions of each step among the
    repeats and replays the block: per-node flip rates are preserved exactly, the
    cross-node / cross-step pairing is destroyed.  If the two arms decode to the
    same final size, an independent-head student is enough and the extra
    correlation machinery has no measured motive.

    h=1 is the control: given (G, x) the frozen policy's flips are conditionally
    independent across nodes, so the shuffle must be a no-op apart from Monte
    Carlo noise.  A stable h=1 difference means the implementation is wrong.

P0-C (archive)
    One teacher trajectory, two archives: every micro step (``full``) and only
    steps 0, h, 2h, ... (``endpoint``).  Both go through the same A.3 decode and
    the same frozen PAVE, and we report the *final solution* size, not the energy,
    because energy and decoded size are not monotone.

    python -u models/diagnose_v8_prior.py --reference_seed 0 --save_dir outputs/v2/none_seed0/checkpoints \
        --result_dir outputs/v8_macro/prior --split validation --graphs 60 \
        --steps 25 50 75 90 --particles 2 --repeats 24 --horizon 2
"""
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from diagnose_v7_census import solution_masks
from eval_v2 import dataset_at, load_policy, sync, write_run
from eval_v7_budget import run_pave
from mis_v1 import adjacency
from mis_v3_search import SearchGraph
from mis_v3_value import PAVEV3
from v7_runs import digest, pave, provenance

BOOTSTRAP = 2000


def build_parser():
    import eval_v3

    parser = eval_v3.build_parser()
    for action in parser._actions:
        if action.dest == "label":
            action.required = False
    parser.set_defaults(split="validation", label="v8_prior", mixed_precision="no",
                        post=["none"], num_t=100, horizon=1)
    parser.add_argument("--reference_seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1017])
    parser.add_argument("--graphs", type=int, default=60)
    parser.add_argument("--steps", type=int, nargs="+", default=[25, 50, 75, 90],
                        help="teacher time points that supply the conditioned state")
    parser.add_argument("--particles", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=24)
    parser.add_argument("--macro_h", type=int, nargs="+", default=[1, 2],
                        help="block lengths; h=1 is the independence control")
    parser.add_argument("--shuffle_seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--phase", choices=("correlation", "archive", "both"), default="both")
    parser.add_argument("--archive_graphs", type=int, default=0,
                        help="graphs used by P0-C; 0 reuses --graphs")
    parser.add_argument("--archive_h", type=int, default=2)
    parser.add_argument("--no_pave", action="store_true",
                        help="skip the frozen PAVE head, report the A.3 solution only")
    return parser


def validate(parser, args):
    expected = dict(num_k=20, num_d=5, beta=1.02, candidate_cap=32, exchange_rounds=3,
                    outer_rounds=8, improving_cap=48, repair_rounds=4, horizon=1)
    for key, value in expected.items():
        if getattr(args, key) != value:
            parser.error(f"the frozen protocol fixes {key}={value}")
    if args.reference_seed != 0:
        parser.error("use --reference_seed 0: PAVE and the backbone are the H1/frozen pair")
    if max(args.steps) >= args.num_t:
        parser.error("conditioned steps must be inside the teacher horizon")
    if args.repeats < 8:
        parser.error("--repeats must be at least 8 for the marginal shuffle to be meaningful")


def teacher_states(solver, graph, adj, seed, steps, particles, normalized):
    """Run the frozen policy and return {step: (N, particles) state} for the steps asked."""
    device = graph.b.device
    generator = torch.Generator(device=device).manual_seed(seed)
    wanted = sorted(set(steps))
    out = {}
    iterator = solver.sample_states(graph, adj, generator=generator, normalized=normalized)
    for step in range(max(wanted) + 1):
        state = next(iterator)
        if step in wanted:
            out[step] = state[:, :particles].clone()
    return out


@torch.no_grad()
def block_replay(solver, graph, adj, x0, h, masks):
    """Replay a recorded flip-mask block from x0 and score every column."""
    problem = solver.problem
    x = x0.clone()
    for step in range(h):
        mask = masks[step]
        x = torch.where(mask, 1 - x, x)
    energy, _ = problem.energy_func(adj, graph.b, x)
    product = adj @ x
    conflicts = 0.5 * (x * product).sum(dim=0)
    decoded = problem.decode_gradient_flip(x, adj, graph.b, return_solutions=True)
    return x, energy, conflicts, decoded.sum(dim=0)


@torch.no_grad()
def sample_block(solver, graph, adj, x0, h, generator, normalized):
    """Draw h steps of the frozen policy from x0, keeping the flip masks."""
    x = x0.clone()
    masks = []
    for _ in range(h):
        logits = solver.flip_logits(x, graph, adj, normalized)
        probability = torch.sigmoid(logits)
        masks.append(torch.rand(probability.shape, device=probability.device,
                                generator=generator) < probability)
        x = torch.where(masks[-1], 1 - x, x)
    return masks


def verify_permutation(masks, permutation):
    """Sanity check: the shuffle must keep each (step, node) flip count."""
    for mask, perm in zip(masks, permutation):
        index = perm.unsqueeze(0).expand(mask.size(0), -1)
        if not torch.equal(mask.sum(dim=1), mask.gather(1, index).sum(dim=1)):
            raise AssertionError("marginal shuffle changed a per-node flip count")
    return True


def correlation_rows(solver, graph, adj, graph_id, args, normalized):
    rows = []
    device = graph.b.device
    states = teacher_states(solver, graph, adj, args.seeds[0], args.steps, args.particles,
                            normalized)
    for step, state in sorted(states.items()):
        for particle in range(state.size(1)):
            x0 = state[:, particle:particle + 1].repeat(1, args.repeats)
            for h in args.macro_h:
                generator = torch.Generator(device=device).manual_seed(
                    args.seeds[0] * 7919 + step * 131 + particle * 17 + h)
                masks = sample_block(solver, graph, adj, x0, h, generator, normalized)
                _, energy, conflicts, sizes = block_replay(solver, graph, adj, x0, h, masks)
                rows.append(dict(graph=graph_id, step=step, particle=particle, h=h, arm="joint",
                                 shuffle_seed=-1,
                                 energy_mean=energy.mean().item(), energy_best=energy.min().item(),
                                 conflicts_mean=conflicts.mean().item(),
                                 size_mean=sizes.float().mean().item(), size_max=int(sizes.max()),
                                 flip_rate=torch.stack(masks).float().mean().item()))
                nodes, repeats = x0.size(0), x0.size(1)
                for shuffle_seed in args.shuffle_seeds:
                    pair = torch.Generator(device=device).manual_seed(
                        shuffle_seed * 104729 + step * 37 + particle)
                    permutation = [torch.randperm(repeats, device=device, generator=pair)
                                   for _ in range(h)]
                    verify_permutation(masks, permutation)
                    shuffled = []
                    for mask, perm in zip(masks, permutation):
                        index = perm.unsqueeze(0).expand(nodes, -1)
                        shuffled.append(mask.gather(1, index))
                    _, s_energy, s_conflicts, s_sizes = block_replay(
                        solver, graph, adj, x0, h, shuffled)
                    rows.append(dict(graph=graph_id, step=step, particle=particle, h=h,
                                     arm="marginal", shuffle_seed=shuffle_seed,
                                     energy_mean=s_energy.mean().item(),
                                     energy_best=s_energy.min().item(),
                                     conflicts_mean=s_conflicts.mean().item(),
                                     size_mean=s_sizes.float().mean().item(),
                                     size_max=int(s_sizes.max()),
                                     flip_rate=torch.stack(shuffled).float().mean().item()))
    return rows


@torch.no_grad()
def archive_rows(solver, graph, adj, graph_id, args, normalized, ranker):
    """P0-C: full micro-step archive versus an endpoint-only archive."""
    problem = solver.problem
    device = graph.b.device
    generator = torch.Generator(device=device).manual_seed(args.seeds[0])
    states = []
    iterator = solver.sample_states(graph, adj, generator=generator, normalized=normalized)
    for step in range(solver.num_t + 1):
        states.append(next(iterator))
    energies = torch.stack([problem.energy_func(adj, graph.b, x)[0] for x in states])  # (T+1, K)
    topology = SearchGraph(graph.num_nodes, graph.edge_index.cpu(), seed=args.seeds[0])
    rows = []
    for name, stride in (("full", 1), ("endpoint", args.archive_h)):
        allowed = [index for index in range(len(states)) if index % stride == 0 or index == len(states) - 1]
        if name == "endpoint" and (len(states) - 1) % stride:
            allowed.append(len(states) - 1)
        picked = energies[allowed]
        best = picked.argmin(dim=0)                       # (K,)
        archived = torch.stack([states[allowed[int(best[k])]][:, k] for k in range(best.numel())],
                               dim=1)
        masks = solution_masks(problem, archived, adj, graph.b)
        best_mask = max(masks, key=int.bit_count)
        row = dict(graph=graph_id, arm=name, stride=stride,
                   arch_energy=float(picked.min(dim=0).values.min()),
                   arch_size=int(best_mask.bit_count()), micro_steps=len(allowed),
                   size_before_pave=int(best_mask.bit_count()))
        if ranker is not None:
            solved = run_pave(topology, best_mask, ranker, args, perf_counter() + 300.0)
            row["size_after_pave"] = int(solved.bit_count())
            row["pave_changed"] = int(solved != best_mask)
        rows.append(row)
    rows.append(dict(graph=graph_id, arm="gap",
                     arch_energy=rows[1]["arch_energy"] - rows[0]["arch_energy"],
                     arch_size=rows[0]["arch_size"] - rows[1]["arch_size"],
                     micro_steps=rows[0]["micro_steps"],
                     size_before_pave=rows[0]["size_before_pave"] - rows[1]["size_before_pave"],
                     size_after_pave=rows[0].get("size_after_pave", rows[0]["size_before_pave"])
                     - rows[1].get("size_after_pave", rows[1]["size_before_pave"])))
    return rows


def bootstrap_ci(values, generator, rounds=BOOTSTRAP):
    if not values:
        return [float("nan")] * 3
    tensor = torch.tensor(values, dtype=torch.float64)
    draws = torch.randint(0, tensor.numel(), (rounds, tensor.numel()), generator=generator)
    means = tensor[draws].mean(dim=1)
    return [float(tensor.mean()), float(means.quantile(0.025)), float(means.quantile(0.975))]


def summarize(rows, args):
    generator = torch.Generator().manual_seed(20261112)
    summary = {}
    correlation = [row for row in rows if "shuffle_seed" in row]
    for h in sorted({row["h"] for row in correlation}):
        block = {}
        for step in sorted({row["step"] for row in correlation}):
            joint = [row for row in correlation if row["h"] == h and row["step"] == step
                     and row["arm"] == "joint"]
            marginal = [row for row in correlation if row["h"] == h and row["step"] == step
                        and row["arm"] == "marginal"]
            if not joint or not marginal:
                continue
            joint_size = mean(row["size_max"] for row in joint)
            marginal_size = mean(row["size_max"] for row in marginal)
            # Per-graph difference with the shuffle arms averaged inside each graph.
            graphs = sorted({row["graph"] for row in joint})
            differences = []
            for graph in graphs:
                left = [row["size_max"] for row in joint if row["graph"] == graph]
                right = [row["size_max"] for row in marginal if row["graph"] == graph]
                if left and right:
                    differences.append(mean(left) - mean(right))
            block[step] = dict(size_joint=joint_size, size_marginal=marginal_size,
                               size_gap=joint_size - marginal_size,
                               size_gap_ci=bootstrap_ci(differences, generator),
                               energy_joint=mean(row["energy_mean"] for row in joint),
                               energy_marginal=mean(row["energy_mean"] for row in marginal),
                               conflicts_joint=mean(row["conflicts_mean"] for row in joint),
                               conflicts_marginal=mean(row["conflicts_mean"] for row in marginal),
                               flip_rate=mean(row["flip_rate"] for row in joint),
                               graphs=len(differences))
        summary[f"h{h}"] = block
    archive = [row for row in rows if row.get("arm") == "gap"]
    if archive:
        for key in ("arch_energy", "arch_size", "size_before_pave", "size_after_pave"):
            values = [row[key] for row in archive if key in row]
            if values:
                summary.setdefault("archive", {})[key] = bootstrap_ci(values, generator)
        summary.setdefault("archive", {})["graphs"] = len(archive)
        summary["archive"]["affected"] = sum(1 for row in archive
                                             if row.get("size_after_pave", row["size_before_pave"]) != 0)
    return summary


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate(parser, args)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "prior"
    output.mkdir(exist_ok=True)
    solver = load_policy(args)
    solver.model.requires_grad_(False)
    ranker = None
    if not args.no_pave:
        checkpoint = torch.load(pave(args.reference_seed), map_location="cpu", weights_only=False)
        ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24))
        ranker.load_state_dict(checkpoint["model_state"])
        ranker = ranker.cpu().eval()
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.graphs)
    rows = []
    started = perf_counter()
    archive_count = args.archive_graphs or len(dataset)
    for index, graph in enumerate(dataset):
        graph_id = Path(dataset.files[index]).stem
        edges = graph.edge_index.cpu()
        graph = graph.to(device)
        adj = adjacency(graph)
        normalized = solver.policy_topology(graph, adj)
        if args.phase in ("correlation", "both") and index < args.graphs:
            rows.extend(correlation_rows(solver, graph, adj, graph_id, args, normalized))
        if args.phase in ("archive", "both") and index < archive_count:
            for row in archive_rows(solver, graph, adj, graph_id, args, normalized, ranker):
                row["edges"] = edges
                rows.append({k: v for k, v in row.items() if k != "edges"})
        sync(device)
        if (index + 1) % 10 == 0:
            print(f"{index + 1}/{len(dataset)} graphs, {perf_counter() - started:.0f}s", flush=True)
    correlation = [row for row in rows if "shuffle_seed" in row]
    archive = [row for row in rows if row.get("arm") in ("full", "endpoint", "gap")]
    if correlation:
        keys = sorted({key for row in correlation for key in row})
        with (output / "correlation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(correlation)
    if archive:
        keys = sorted({key for row in archive for key in row})
        with (output / "archive.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(archive)
    summary = summarize(rows, args)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    write_run(args.result_dir, dict(version="v8_macro", status="completed", mode="prior",
                                    graphs=args.graphs, steps=args.steps, repeats=args.repeats,
                                    macro_h=args.macro_h, shuffle_seeds=args.shuffle_seeds,
                                    pave=ranker is not None,
                                    backbone_sha256=digest(Path(args.save_dir) / "best.pt"),
                                    summary=summary,
                                    provenance=provenance(args.reference_seed),
                                    seconds=perf_counter() - started,
                                    config={k: str(v) for k, v in vars(args).items()}))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
