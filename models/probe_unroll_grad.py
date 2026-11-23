"""v11 C3 gradient self-proof (read-only probe, direction §2.9).

Five checks on 20 validation graphs, K=20, 40 unroll steps, decode every 10:

1. finiteness  -- loss / logits finite, gradient free of NaN/Inf;
2. non-zero    -- ``|g|_1 > 0`` and ``std(g)/mean(|g|) > 1e-3``;
3. direction   -- a small step along ``-g`` (eta = 1e-3*|logits|/|g|) must *increase*
                  the decoded size on average, ``+g`` must decrease it;
4. correlation -- Spearman(|g|, delta size) > 0.2 over the 20 graphs;
5. scale       -- ``|g|/|logits|`` within [1e-4, 1e2].

Checks 1-4 must pass before A1 may be launched (direction §2.9).  The probe never
writes model weights; it only reads the checkpoint named by ``--save_dir``.

Every replay rebuilds its ``torch.Generator`` from the same per-graph seed, mirroring
``PaddedSearch.load(graph, edges, seed)`` in deployment.  Two extra columns make the
direction check falsifiable: ``delta_control`` (random perturbation of the same norm --
the chaos floor) and ``determinism`` (a repeated identical replay must be bit-identical).

Usage (server, GPU1):
    CUDA_VISIBLE_DEVICES=1 python models/probe_unroll_grad.py \
        --save_dir outputs/v2/none_seed0/checkpoints \
        --out outputs/v11/probe_unroll_grad.json
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from eval_v2 import dataset_at, load_policy
from main import build_parser
from mis_v1 import adjacency
from mis_v11_unroll import decode_size, generator_for, unroll, unroll_objective


def probe_parser():
    parser = build_parser()
    parser.set_defaults(method='RLNN', test_path=None, mixed_precision='fp16', seed=1017,
                        num_k=20, num_t=100, num_l=5, num_h=128, num_d=5, num_workers=0,
                        save_dir='outputs/v2/none_seed0/checkpoints')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--graphs', type=int, default=20)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--decode_every', type=int, default=10)
    parser.add_argument('--probe_seed', type=int, default=2027)
    parser.add_argument('--out', type=Path, default=Path('outputs/v11/probe_unroll_grad.json'))
    return parser


def spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or left.std() == 0 or right.std() == 0:
        return 0.0
    rank_l = np.argsort(np.argsort(left)).astype(np.float64)
    rank_r = np.argsort(np.argsort(right)).astype(np.float64)
    return float(np.corrcoef(rank_l, rank_r)[0, 1])


def feasible(decoded, edge_index):
    """Independent-set assertion on every decoded column."""
    selected = decoded.bool()
    return not bool((selected[edge_index[0]] & selected[edge_index[1]]).any())


def run_probe(args):
    solver = load_policy(args)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.graphs)
    rows = []
    started = perf_counter()

    def rollout(index, graph, adj, topology, *, logits_bias=None, keep_grad=False):
        """One replay from a *freshly reseeded* generator.

        Every rollout must rebuild its generator from the same seed, exactly like the
        deployed engine re-seeds in ``PaddedSearch.load(graph, edges, seed)``.  Reusing
        one advancing ``torch.Generator`` object across replays would feed a different
        noise stream, so ``+-grad`` differences would be noise, not gradient effect.
        """
        gen = generator_for(device, args.probe_seed + index)
        x0 = torch.randint(0, 2, (graph.num_nodes, args.num_k), device=device,
                           dtype=solver.dtype, generator=gen)
        if logits_bias is None and keep_grad:
            return unroll_objective(solver, graph, adj, topology, x0, steps=args.steps,
                                    decode_every=args.decode_every, generator=gen), x0
        return unroll(solver, graph, adj, topology, x0, steps=args.steps,
                      decode_every=args.decode_every, generator=gen,
                      logits_bias=logits_bias, keep_grad=keep_grad), x0

    for index in range(len(dataset)):
        graph = dataset[index].to(device)
        adj = adjacency(graph)
        topology = solver.policy_topology(graph, adj)
        out, x0 = rollout(index, graph, adj, topology, keep_grad=True)
        loss = out['loss']
        logits = out['logits']
        grad, = torch.autograd.grad(loss, logits, retain_graph=False, allow_unused=False)

        logits_norm = float(logits.detach().float().norm())
        grad_norm = float(grad.detach().float().norm())
        eta = 1e-3 * logits_norm / max(grad_norm, 1e-12)

        base_size = float(out['sizes'][-1])
        minus, _ = rollout(index, graph, adj, topology, logits_bias=-eta * grad.detach())
        plus, _ = rollout(index, graph, adj, topology, logits_bias=eta * grad.detach())
        # control: a random perturbation of the *same* norm (chaos floor for check 3)
        noise = torch.randn_like(grad.detach())
        noise = noise / noise.float().norm().clamp_min(1e-12) * grad_norm
        control, _ = rollout(index, graph, adj, topology, logits_bias=eta * noise)
        # determinism floor: identical seed and identical perturbation -> bit-identical
        repeat, _ = rollout(index, graph, adj, topology, logits_bias=-eta * grad.detach())
        finite = bool(torch.isfinite(loss).all() and torch.isfinite(logits).all()
                      and torch.isfinite(grad).all())
        rows.append(dict(
            graph_id=str(index),
            nodes=int(graph.num_nodes),
            loss=float(loss.detach()),
            base_size=round(base_size, 4),
            size_minus_grad=round(float(minus['sizes'][-1]), 4),
            size_plus_grad=round(float(plus['sizes'][-1]), 4),
            delta_minus=round(float(minus['sizes'][-1]) - base_size, 6),
            delta_plus=round(float(plus['sizes'][-1]) - base_size, 6),
            delta_control=round(float(control['sizes'][-1]) - base_size, 6),
            determinism=abs(float(repeat['sizes'][-1]) - float(minus['sizes'][-1])) < 1e-9,
            logits_norm=logits_norm,
            grad_norm=grad_norm,
            grad_std_over_absmean=float(grad.detach().float().std()
                                        / grad.detach().float().abs().mean().clamp_min(1e-12)),
            eta=eta,
            finite=finite,
            feasible=feasible(
                decode_size(solver.problem, out['state'].detach(), adj, graph.b)[1],
                graph.edge_index),
        ))

    delta_minus = float(np.mean([row['delta_minus'] for row in rows]))
    delta_plus = float(np.mean([row['delta_plus'] for row in rows]))
    delta_control = float(np.mean([row['delta_control'] for row in rows]))
    determinism = all(row['determinism'] for row in rows)
    grad_norms = [row['grad_norm'] for row in rows]
    deltas = [row['delta_minus'] for row in rows]
    rho = spearman(grad_norms, deltas)
    ratio = [row['grad_norm'] / max(row['logits_norm'], 1e-12) for row in rows]
    checks = {
        'finite': all(row['finite'] for row in rows),
        'nonzero_grad': all(row['grad_norm'] > 0
                            and row['grad_std_over_absmean'] > 1e-3 for row in rows),
        'direction': delta_minus > 0 and delta_plus < 0,
        'correlation': rho > 0.2,
        'scale': all(1e-4 <= value <= 1e2 for value in ratio),
        'feasible': all(row["feasible"] for row in rows),
        'determinism': determinism,
    }
    report = dict(
        script='models/probe_unroll_grad.py',
        checkpoint=str(args.save_dir),
        graphs=len(rows),
        steps=args.steps,
        decode_every=args.decode_every,
        num_k=args.num_k,
        probe_seed=args.probe_seed,
        split=args.split,
        test_path=str(args.test_path),
        checks=checks,
        required=('finite', 'nonzero_grad', 'direction', 'correlation', 'feasible'),
        passed=all(checks[key] for key in ('finite', 'nonzero_grad', 'direction',
                                           'correlation', 'feasible')),
        summary=dict(delta_minus=round(delta_minus, 6), delta_plus=round(delta_plus, 6),
                     delta_control=round(delta_control, 6),
                     grad_minus_over_control=round(
                         (delta_minus - delta_control) / max(abs(delta_control), 1e-9), 4),
                     determinism=determinism,
                     spearman_grad_vs_delta=round(rho, 4),
                     grad_over_logits=[round(value, 6) for value in ratio[:5]],
                     grad_norm_mean=round(float(np.mean(grad_norms)), 6)),
        wall_s=round(perf_counter() - started, 3),
        rows=rows,
    )
    return report


def main():
    args = probe_parser().parse_args()
    report = run_probe(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('checks', 'summary', 'passed', 'wall_s')},
                     indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    if not report['passed']:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
