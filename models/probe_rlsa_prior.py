"""Neural-guided RLSA: keep the annealer, add the frozen policy as an energy bias.

Two facts frame this experiment.

* RLSA is the strongest solver at a one-second budget (SIZE 20.073 at 0.65 s,
  20.093 at 1.29 s, measured in ``outputs/v8_macro/rlsa_curve_k200``), and it has
  no learning in it at all.
* Our frozen sampler saturates at 19.983 at 1.0 s and does not move when the step
  count, the particle width or the operator is changed.

So the learned part is not strong enough to be the searcher, but the annealer is
strong enough to be one.  The composition nobody has tried here: run RLSA's
kth-value update and add the frozen policy's logit as an extra term of the energy
gradient, i.e.

    delta = grad * (2x - 1) / 2  +  w * s * logit_i

with the logits recomputed every ``--prior_every`` steps (a per-step GCN pass at
``num_k=200`` costs ~1.3 ms against RLSA's 0.59 ms, so the prior has to be stale
by a few steps to stay affordable) and ``s = std(delta) / std(logit)`` so that
``w`` reads as the fraction of delta's spread the prior is allowed to contribute.

The probe reports SIZE against wall time for several weights, so a weight can be
kept only if it pays for the extra GCN time.
"""
import csv
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_rlsa_curve import curve_parser
from eval_v2 import TEST, TEST_CACHE, dataset_at, sync
from mis_v3_search import SearchGraph
from mis_v7_recombine import is_independent
from models import GCN, edge2adj
from solvers import RLSA

DEFAULT_POLICY = Path("/home/hzeng/project/ZH/RLNN_lab/outputs/v2/none_seed0/checkpoints")


@torch.no_grad()
def run(solver, policy, graph, num_t, seed, prior_weight, prior_every, normalise):
    """RLSA with an optional learned bias; returns (integer mask, wall seconds)."""
    generator = torch.Generator(device=graph.b.device).manual_seed(seed)
    x = torch.randint(0, 2, (graph.num_nodes, solver.num_k),
                      device=solver.accelerator.device,
                      dtype=graph.edge_weight.dtype, generator=generator)
    A = torch.sparse_coo_tensor(
        graph.edge_index, graph.edge_weight,
        torch.Size((graph.num_nodes, graph.num_nodes)),
    ).to_sparse_csr()
    normalised = edge2adj(graph.edge_index, graph.num_nodes)
    energy, grad = solver.problem.energy_func(A, graph.b, x, True)
    best_sol, best_energy = x.clone(), energy.clone()
    bias, scale = None, 1.0
    started = perf_counter()
    for epoch in range(num_t):
        tau = solver.tau0 * (1 - epoch / num_t)
        delta = grad * (2 * x - 1) / 2
        if prior_weight and (epoch % prior_every == 0 or bias is None):
            logit = policy(x.unsqueeze(-1), graph.edge_index, normalised).squeeze(-1).detach()
            if normalise:
                # express the bias in delta's own unit: w = fraction of delta's spread
                scale = float(delta.std().clamp_min(1e-6) / logit.std().clamp_min(1e-6))
            bias = logit
        if bias is not None:
            delta = delta + prior_weight * scale * bias
        term2 = -torch.kthvalue(-delta, solver.num_d, dim=0, keepdim=True).values
        flip_prob = torch.sigmoid((delta - term2) / tau)
        rr = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        x = torch.where(rr < flip_prob, 1 - x, x)
        energy, grad = solver.problem.energy_func(A, graph.b, x, True)
        improves = energy < best_energy
        best_sol = torch.where(improves.unsqueeze(0).repeat(graph.num_nodes, 1), x, best_sol)
        best_energy = torch.where(improves, energy, best_energy)
    sync(graph.b.device)
    seconds = perf_counter() - started
    solutions = solver.problem.decode_gradient_flip(best_sol, A, graph.b, return_solutions=True)
    best = solutions[:, solutions.sum(dim=0).argmax()]
    return int(sum(1 << int(node) for node in best.nonzero().flatten().tolist())), seconds


def main():
    parser = curve_parser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--policy_dir", default=str(DEFAULT_POLICY))
    parser.add_argument("--policy_h", type=int, default=128)
    parser.add_argument("--policy_l", type=int, default=5)
    parser.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--prior_every", type=int, default=10)
    parser.add_argument("--no_normalise", dest="normalise", action="store_false",
                        help="add the raw logit instead of rescaling it to delta's spread")
    args = parser.parse_args()
    if args.test_path is None:
        args.test_path = str(TEST_CACHE if (TEST_CACHE / "tensor_manifest.json").is_file() else TEST)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    solver = RLSA(args)
    solver.prepare()
    device = solver.accelerator.device

    policy = GCN(1, args.policy_h, args.policy_l, 1).to(device)
    state = torch.load(Path(args.policy_dir) / "best.pt", map_location=device, weights_only=True)
    policy.load_state_dict(state, strict=False)
    policy.eval()
    print(f"policy: GCN(1,{args.policy_h},{args.policy_l},1) from {args.policy_dir}", flush=True)

    dataset = dataset_at(args.test_path, solver.dtype, args.samples, official=args.split == "test")
    seed = args.seeds[0]
    rows = []
    for index in range(len(dataset)):
        edges = dataset[index].edge_index
        graph = dataset[index].to(device)
        neighbours = SearchGraph(int(graph.num_nodes), edges, seed=seed).neighbors
        for weight in args.weights:
            mask, seconds = run(solver, policy, graph, args.num_t, seed, weight,
                                args.prior_every, args.normalise)
            if not is_independent(mask, neighbours):
                raise SystemExit(f"illegal solution: graph {index}, weight {weight}")
            rows.append(dict(graph_id=f"idx_{index}", weight=weight, num_t=args.num_t,
                             num_k=args.num_k, size=mask.bit_count(),
                             wall_s=round(seconds, 4)))
        if (index + 1) % 20 == 0:
            parts = [f"w={w}:{mean(r['size'] for r in rows if r['weight'] == w):.3f}"
                     f"@{mean(r['wall_s'] for r in rows if r['weight'] == w):.3f}s"
                     for w in args.weights]
            print(f"{index + 1}/{len(dataset)} " + " ".join(parts), flush=True)

    path = args.result_dir / f"rlsa_prior_n{args.num_k}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for weight in args.weights:
        chosen = [row for row in rows if row["weight"] == weight]
        summary[str(weight)] = dict(size=mean(r["size"] for r in chosen),
                                    wall=mean(r["wall_s"] for r in chosen))
    (args.result_dir / f"rlsa_prior_n{args.num_k}.json").write_text(
        json.dumps(dict(num_t=args.num_t, num_k=args.num_k, num_d=args.num_d,
                        prior_every=args.prior_every, normalise=args.normalise,
                        samples=len(dataset), summary=summary), indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
