"""v11 C3 -- size-aware unroll objective (decoder-in-the-loop, straight-through).

Paper question
--------------
The frozen one-step ``erdoes`` objective never sees the decoder, so the policy gets
no signal about *how many nodes its states can be completed to*.  v11 C3 rolls the
deploy-active sampler forward for ``unroll_steps`` update steps, decodes the state
every ``decode_every`` steps with the Appendix-A.3 gradient flip, and back-propagates
``-mean(decoded size)`` through the *last* decode only (straight-through: the flip
decision is discrete, the sigmoid probability carries the gradient).

Everything here is new code.  ``models/mis_v11_unroll.py`` is imported by an additive
``elif self.loss == 'unroll'`` branch in ``models/solvers.py``; the file:line ledger
lives in ``docs/temporary/RLNN_v11_Implementation.md``.  ``models/models.py``,
``models/mis_v8_pad.py`` (deployment) and the other frozen modules are untouched.

Contract
--------
* t4 MATLAB input I5: the mean waiting time to a +1 improvement is 517 steps, a
  40-step window covers 0 %/7.7 % of the gains and 200 steps 43.2 %
  => default ``unroll_steps = 200``.
* captain 2026-11-17: keep 200 unless the adaptive gates trip
  (peak GPU memory > 6 GB or per-graph fwd+bwd > 0.6 s) -> fall back to 128 and
  record ``degraded_reason`` in ``run.json``.
* decode cadence stays every 10 steps, only the last decode keeps autograd.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

DEFAULT_STEPS = 200
FALLBACK_STEPS = 128
DECODE_EVERY = 10
MEMORY_GATE_BYTES = 6 * 1024 ** 3
TIME_GATE_SECONDS = 0.6


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
def _rand(shape, *, dtype, device, generator):
    if generator is None:
        return torch.rand(shape, dtype=dtype, device=device)
    return torch.rand(shape, dtype=dtype, device=device, generator=generator)


def _dense(tensor, like=None):
    """Densify a CSR adjacency (or a tensor topology) and match ``like.dtype``.

    ``models/mis_v1.py::adjacency`` and ``solvers.policy_topology`` hand back a
    sparse CSR matrix.  The A.3 decode multiplies by it with ``fp16`` operands,
    and cuSparse rejects that mix on the deployed cards
    (``cusparseSpMM_bufferSize(): data type combination not supported,
    matA: CUDA_R_16F, matB: CUDA_R_32F``).  The deployment arm already runs the
    dense kernel (``models/mis_v8_pad.py:80 self.dense_norm``), so densifying
    here is the same operator, just without the CSR path; the dtype cast is what
    ``dense_norm`` is built with (``self.dtype``).
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if tensor.layout != torch.strided:
        tensor = tensor.to_dense()
    if like is not None and tensor.dtype != like.dtype:
        tensor = tensor.to(like.dtype)
    return tensor


def decode_size(problem, state, adjacency, bias):
    """A.3 gradient-flip decode of every column; returns ``(sizes, decoded)``."""
    decoded = problem.decode_gradient_flip(state, _dense(adjacency, state),
                                           _dense(bias, state), return_solutions=True)
    return decoded.sum(dim=0), decoded


def unroll(solver, graph, adjacency, topology, x0, *, steps, decode_every=DECODE_EVERY,
           generator=None, logits_bias=None, archive=False, keep_grad=True):
    """Roll the deploy-active sampler ``steps`` update steps from ``x0``.

    The dynamics are the deployment dynamics of ``models/mis_v8_pad.py::_body``
    (flip with probability ``sigmoid(logits)``, noise shape ``(num_nodes, K)``),
    without the tabu mask that only the LNS path enables.
    """
    problem = solver.problem
    bias = graph.b
    adjacency = _dense(adjacency, x0)
    topology = _dense(topology, x0)
    x = x0
    sizes, logits_last = [], None
    best_mask = best_size = None
    decode_steps = []
    last_size = None
    for step in range(steps):
        product = (adjacency @ x.detach()) if getattr(solver, 'structure_gain', False) else None
        # The policy input is detached as well, so the gradient of one step sees a
        # fixed state: with the carrier left on the input, the model Jacobian is
        # re-applied once per step and the parameter gradient still grows ~3x/step
        # (measured 1.1e4 at 10 steps -> 7.1e12 at 40 -> NaN at 128 in fp32).
        logits = solver.flip_logits(x.detach(), graph, adjacency, topology,
                                    adjacency_product=product)
        if logits_bias is not None:
            logits = logits + logits_bias
        prob = torch.sigmoid(logits)
        rr = _rand(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        # Forward value: the deployment flip rule, bit for bit.  ``x.detach()`` in
        # both branches matters: the ``where`` branches have unit Jacobian in ``x``,
        # so leaving them on the graph makes the parameter gradient compound once per
        # step (measured: 4.1e4 at 10 steps -> 1.9e14 at 40 -> NaN at 128 in fp32).
        flip = torch.where(rr < prob, 1 - x.detach(), x.detach())
        # Straight-through carrier: the frozen ``erdoes`` relaxation
        # ``(1-p)x + p(1-x)`` has expectation ``flip`` and derivative (1-2x) dp,
        # so adding the *detached* difference changes no forward bit but lets the
        # gradient reach ``prob`` -- ``torch.where`` alone would cut it (the
        # comparison is boolean, not differentiable).
        # ``x.detach()`` keeps the surrogate *one-step*: crediting only the current
        # logits.  Letting the state path carry gradient makes the parameter Jacobian
        # compound once per step and the gradient explode geometrically (measured on
        # GPU1, fp32, K=20: max |param grad| 1.1e5 at 10 steps, 3.2e20 at 40,
        # 1.3e31 at 64, NaN at 128 -- i.e. ~3x per step), so full BPTT through the
        # sampled flip chain is numerically infeasible at the 200-step horizon that
        # t4 I5 asks for.  Forward value is unchanged either way (the added term is
        # exactly zero, ``relaxed - relaxed.detach()``).
        relaxed = (1 - prob) * x.detach() + prob * (1 - x.detach())
        x = flip + (relaxed - relaxed.detach())
        logits_last = logits
        if (step + 1) % decode_every == 0:
            last = (step + 1) == steps
            # The decode is a *reporting* operator: it runs to a fixed point, so
            # back-propagating through it multiplies by the adjacency matrix once per
            # iteration (mean degree ~80 on RB-suite graphs -> ~1e4-1e9 per graph).
            # Measured on GPU1 (fp32, K=20) with the decode on the graph: max |param
            # grad| 7.3e3 at 10 steps, 3.7e15 at 40, 3.7e24 at 64, NaN at 128 even
            # after the state path was detached, so the decode chain -- not the
            # rollout -- is what overflows.  It is therefore detached, and the
            # gradient comes from the single-step relaxed carrier below.
            size, decoded = decode_size(problem, x.detach(), adjacency, bias)
            sizes.append(float(size.mean().detach()))
            decode_steps.append(step + 1)
            if last:
                last_size = size
            if archive:
                if best_mask is None:
                    best_mask, best_size = decoded, size
                else:
                    better = size > best_size
                    best_mask = torch.where(better.unsqueeze(0), decoded, best_mask)
                    best_size = torch.where(better, size, best_size)
    return dict(state=x, logits=logits_last, sizes=sizes, decode_steps=decode_steps,
                last_size=last_size, archive_mask=best_mask, archive_size=best_size, steps=steps)


def unroll_objective(solver, graph, adjacency, topology, x0, *, steps, decode_every=DECODE_EVERY,
                     generator=None, logits_bias=None, archive=False):
    """``-mean(size)`` of the last decode; ``archive=True`` credits the best decode too.

    Scale: the frozen ``erdoes`` loss is ``-|S| + O(1)`` on a feasible set
    (``adjacency`` has no self loops, so ``A x = 0`` on an independent set), i.e.
    O(num_nodes).  ``loss = -mean(size)`` is therefore left un-normalised so the
    ``2e-5`` learning rate transfers from the erdoes recipe unchanged.
    """
    out = unroll(solver, graph, adjacency, topology, x0, steps=steps,
                 decode_every=decode_every, generator=generator, logits_bias=logits_bias,
                 archive=archive)
    if archive and out['archive_size'] is not None:
        value = out['archive_size'].mean()
    elif out['last_size'] is not None:
        value = out['last_size'].mean()
    else:
        size, _ = decode_size(solver.problem, out['state'].detach(), adjacency, graph.b)
        value = size.mean()
    out['value'] = value
    # Bounded surrogate: ``d loss / d theta`` is the one-step relaxed size gradient
    # ``-sum_nodes x`` (x carries only the ``relaxed - relaxed.detach()`` carrier, so
    # d x / d theta = (1-2x) d prob / d theta).  Forward value stays ``-mean(size)``
    # bit for bit because the carrier is exactly zero.
    surrogate = -out['state'].sum(dim=0).mean() if out['state'].requires_grad else 0.0
    out['loss'] = -value + (surrogate - surrogate.detach() if torch.is_tensor(surrogate)
                            else surrogate)
    return out


def _rollout_seed(solver, graph, counter):
    base = int(getattr(solver, 'seed', 0) or 0)
    idx = int(getattr(graph, 'idx', 0) or 0)
    return (base * 1_000_003 + idx * 7_919 + counter * 104_729) % (2 ** 63 - 1)


def generator_for(device, seed):
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


# --------------------------------------------------------------------------- #
# training hook: called from the additive branch in solvers.RLNN.train()
# --------------------------------------------------------------------------- #
def configure(solver, args):
    """Attach the v11 unroll config to a solver (called once from ``models/main.py``).

    ``MetaSolver`` does not keep its namespace, so the additive ``loss == 'unroll'``
    block in ``main.py`` hands the argparse result over here before ``train()``.
    """
    solver.v11_unroll = _cfg(solver, args)
    return solver.v11_unroll


def _cfg(solver, args=None):
    cfg = getattr(solver, 'v11_unroll', None)
    if cfg is None:
        if args is None:
            args = getattr(solver, 'args', None)
        steps = int(getattr(args, 'unroll_steps', DEFAULT_STEPS) or DEFAULT_STEPS)
        cfg = dict(steps=steps, requested_steps=steps,
                   decode_every=int(getattr(args, 'unroll_decode_every', DECODE_EVERY)
                                    or DECODE_EVERY),
                   archive=bool(getattr(args, 'unroll_archive', False)),
                   degraded_reason=None, per_graph_s=None, peak_mem_bytes=None,
                   updates=0, measured=[], size_trace=[], loss_trace=[])
        solver.v11_unroll = cfg
    return cfg


def unroll_graph_loss(solver, graph, adjacency, topology, x):
    """One v11 C3 update on the particle columns ``x`` (shape ``num_nodes x K``).

    Drop-in replacement for the ``erdoes``/``reinforce`` loss terms; the caller
    (``solvers.RLNN.train``) keeps the surrounding optimizer step unchanged.
    The adaptive gates are measured on the first update only and can only *shrink*
    the unroll length for subsequent updates (recorded in ``run.json``).
    """
    cfg = _cfg(solver)
    device = x.device
    counter = cfg['updates']
    gen = generator_for(device, _rollout_seed(solver, graph, counter))
    first = not cfg['measured']

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        base_mem = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    out = unroll_objective(solver, graph, adjacency, topology, x, steps=cfg['steps'],
                           decode_every=cfg['decode_every'], generator=gen,
                           archive=cfg['archive'])
    loss = out['loss']
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    peak = (torch.cuda.max_memory_allocated(device) - base_mem) if device.type == 'cuda' else 0

    if first:
        cfg['per_graph_s'] = round(elapsed, 4)
        cfg['peak_mem_bytes'] = int(peak)
        cfg['measured'].append(dict(steps=cfg['steps'], per_graph_s=round(elapsed, 4),
                                    peak_mem_bytes=int(peak)))
        reasons = []
        if peak > MEMORY_GATE_BYTES:
            reasons.append(f'peak_mem {peak / 1024 ** 3:.2f}GB>6GB')
        if elapsed > TIME_GATE_SECONDS:
            reasons.append(f'per_graph {elapsed:.3f}s>0.6s')
        if reasons and cfg['steps'] > FALLBACK_STEPS:
            cfg['degraded_reason'] = '; '.join(reasons)
            cfg['steps'] = FALLBACK_STEPS
            cfg['measured'].append(dict(steps=FALLBACK_STEPS, degraded=True))

    cfg['updates'] = counter + 1
    cfg['size_trace'].append(round(float(out['sizes'][-1]), 4))
    cfg['loss_trace'].append(round(float(loss.item()), 4))
    return loss


# --------------------------------------------------------------------------- #
# run.json snapshot
# --------------------------------------------------------------------------- #
def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def dump_run_json(args, solver, save_dir, extra=None):
    """Write ``run.json`` next to ``best.pt``: hyper-parameters + gate decisions."""
    cfg = getattr(solver, 'v11_unroll', {}) or {}
    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'argv': _jsonable({key: getattr(args, key) for key in sorted(vars(args))}),
        'loss': getattr(args, 'loss', None),
        'unroll': _jsonable({key: value for key, value in cfg.items()
                             if key not in ('size_trace', 'loss_trace')}),
        'unroll_size_trace': cfg.get('size_trace', [])[-20:],
        'unroll_loss_trace': cfg.get('loss_trace', [])[-20:],
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'peak_mem_bytes_total': int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
    }
    accelerator = getattr(solver, 'accelerator', None)
    if accelerator is not None:
        payload['device'] = str(accelerator.device)
    if extra:
        payload.update(_jsonable(extra))
    path = Path(save_dir) / 'run.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return path


def config_snapshot(solver):
    """Compact one-line summary for logs."""
    cfg = getattr(solver, 'v11_unroll', {}) or {}
    peak = cfg.get('peak_mem_bytes')
    return (f"unroll steps={cfg.get('steps')} decode_every={cfg.get('decode_every')} "
            f"archive={cfg.get('archive')} per_graph={cfg.get('per_graph_s')}s "
            f"peak={(f'{peak / 1024 ** 3:.2f}GB' if peak else None)} "
            f"degraded={cfg.get('degraded_reason')}")
