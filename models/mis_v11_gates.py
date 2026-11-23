"""v11 hard gates (four assertions the captain requires before A1).

G-A  feasible solutions      -- every decoded column is an independent set, and the
                                A.3 decode never shrinks a state (monotone).
G-B  reference equivalence   -- (i) the captured engine reproduces the un-captured
                                ``eager_step`` path bit for bit (same seed, same
                                steps); (ii) the training-side unroll stepper
                                reproduces one engine block bit for bit;
                                (iii) the captured decoder agrees with the
                                reference A.3 decode on the decoded *size*.
G-C  deadline semantics      -- ``run`` syncs after every block and the wall clock
                                never overshoots the budget by more than one block;
                                ``max_blocks`` gives exactly ``chunk*blocks`` steps
                                (the semantics ``--fixed_steps`` relies on).
G-D  same-seed reproducibility -- two identical runs return identical masks and
                                states, bit for bit.

Modes
-----
``--selftest``  CPU only, no checkpoint, no dataset: exercises ``models/mis_v11_unroll.py``
                (straight-through gradient, monotone decode, feasibility, seeded
                determinism, adaptive-gate degrade path).
``--engine``    GPU: G-A..G-D on the real checkpoint + the first 4 author graphs.
``--gate0``     reads the ``eval_v8_padded`` output of A0 and checks the dual-caliber
                gate 0 (t11 / captain R3, R7): the only judgement column is
                ``search_size`` against the **declared-config** A0 (20.0690: fp16 +
                dense, 810 fixed steps) with |delta| <= 0.01; the fp32<->fp16
                configuration sensitivity 0.007 is context only, never a tolerance; the
                step count must be exactly 810; dtype/dense/fp32_csr are read from the
                ``--v11_csv`` artefact (100 % row consistency), never from the flags.
                ``final_size`` is reported only (+-0.0005 PAVE reserve jitter).
                ``--negative-selftest`` corrupts the artefact (+0.05 shift, fp16->fp32)
                and requires the gate to fail.

Every mode writes ``outputs/v11/*.json`` and exits non-zero when a gate is red.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

import mis_v11_unroll as V11

REFERENCE_SOLUTION = 20.0685
REFERENCE_TOLERANCE = 0.0005
STEP_RANGE = (700, 860)
FIXED_STEPS = 808
FP32_CSR_MS = 3.9
FP16_DENSE_MS = (1.05, 1.30)

# --- t11 (captain 2026-11-17, R3/R7) --------------------------------------- #
# A0 is the *current build* at a fixed 810 steps (808 requested -> 81 blocks of
# chunk 10).  ``search_size`` is the only judgement column: it is the search-only
# number, while ``final_size`` carries the PAVE 0.2 s wall-clock reserve and moves
# by +-0.0005 between two runs of the same command.
# 20.0685 (outputs/v8_macro/arm_fp16_dense) belongs to an old run whose exact
# configuration is not recoverable -- see RLNN_v11_Implementation.md §7.4 -- so it
# is kept as a historical clock-caliber reference only, never as a gate target.
#
# t16 (captain 2026-11-17, t13-F1): A0 is the **declared config** (fp16 + dense,
# 810 fixed steps) = 20.0690.  It used to be 20.0760, which is the fp32 MISCONFIG
# run (``--mixed_precision`` not passed).  A reference biased by +0.007 combined
# with a +-0.01 tolerance gives the acceptance band [20.0660, 20.0860] where it
# should be [20.0590, 20.0790] -- i.e. a real regression anywhere in 20.080-20.086
# would have passed.  The bias is a deterministic configuration difference, not
# run-to-run jitter, so it can neither be averaged away nor absorbed by tolerance.
REFERENCE_SEARCH = 20.0690            # A0 = declared config (fp16+dense), 810 steps
REFERENCE_SEARCH_TOLERANCE = 0.01     # absolute band on mean search_size
CONFIG_SENSITIVITY = 0.007            # fp32<->fp16; narration only, NOT in the tolerance
HISTORICAL_CLOCK_REFERENCE = 20.0685  # old CSV: historical_clock_reference_unreproducible
FINAL_SIZE_JITTER = 0.0005            # PAVE reserve jitter, final_size only
FIXED_STEPS_RUN = 810                 # ceil(808 / 10) * 10


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def unpack(bits, nodes):
    return torch.tensor([(int(bits) >> index) & 1 for index in range(nodes)], dtype=torch.bool)


def independent(mask, edge_index):
    """True when no edge has both endpoints selected (``edge_index`` is 2 x E)."""
    if mask.numel() == 0 or edge_index.numel() == 0:
        return True
    return not bool((mask[edge_index[0]] & mask[edge_index[1]]).any())


def reference_problem():
    """The frozen A.3 decoder when torch_geometric is importable, else a copy."""
    try:
        from problems.mis import Problem
        return Problem(1.02), 'problems.mis.Problem'
    except Exception:  # local CPU box without torch_geometric
        class _Copy:
            def __init__(self, beta):
                self.beta = beta

            def decode_gradient_flip(self, solutions, adjacency, bias, *, return_solutions=False):
                solutions = solutions.clone()
                particles = torch.arange(solutions.size(1), device=solutions.device)
                while True:
                    gradient = 2 * self.beta * (adjacency @ solutions) + bias
                    delta = gradient * (2 * solutions - 1) / 2
                    gain, node = delta.max(dim=0)
                    improves = gain > 0
                    if not improves.any():
                        return solutions if return_solutions else -solutions.sum(dim=0)
                    selected = solutions[node, particles]
                    solutions[node, particles] = torch.where(improves, 1 - selected, selected)
        return _Copy(1.02), 'inline copy of problems.mis.Problem'


class StubSolver:
    """Minimal stand-in for ``solvers.RLNN`` on the CPU (selftest only)."""

    def __init__(self, problem, nodes, num_k=5, seed=0, steps=20, decode_every=5):
        torch.manual_seed(3)
        self.problem = problem
        self.structure_gain = False
        self.num_k = num_k
        self.dtype = torch.float32
        self.seed = seed
        self.hidden = 8
        self.left = (torch.randn(self.hidden, nodes, dtype=self.dtype) * 0.1).requires_grad_(True)
        self.right = (torch.randn(nodes, self.hidden, dtype=self.dtype) * 0.1).requires_grad_(True)
        self.args = SimpleNamespace(unroll_steps=steps, unroll_decode_every=decode_every,
                                    unroll_archive=False)

    def flip_logits(self, x, graph, adjacency, topology, adjacency_product=None):
        product = adjacency @ x
        return self.right @ torch.tanh(self.left @ product)


def build_cpu_graph(nodes=30, degree=4, seed=0):
    rng = np.random.default_rng(seed)
    pairs = set()
    for node in range(nodes):
        for _ in range(degree):
            other = int(rng.integers(0, nodes))
            if other != node:
                pairs.add((min(node, other), max(node, other)))
    edges = sorted(pairs)
    edge_index = torch.tensor([[a, b] for a, b in edges] + [[b, a] for a, b in edges],
                              dtype=torch.long).reshape(2, -1)
    graph = SimpleNamespace(num_nodes=nodes, idx=0,
                            edge_index=edge_index,
                            edge_weight=torch.ones(edge_index.shape[1], dtype=torch.float32) / 2,
                            b=-torch.ones(nodes, 1, dtype=torch.float32))
    adjacency = torch.sparse_coo_tensor(edge_index, graph.edge_weight,
                                        (nodes, nodes)).to_sparse_csr()
    return graph, adjacency, edge_index


def greedy_mis(nodes, edge_index):
    """Smallest-degree-first greedy MIS: the start the tabu probe walks from."""
    neighbours = [0] * nodes
    degree = np.zeros(nodes, dtype=np.int64)
    for left, right in edge_index.t().tolist():
        neighbours[left] |= 1 << right
        neighbours[right] |= 1 << left
        degree[left] += 1
        degree[right] += 1
    mask = 0
    for node in np.argsort(degree, kind='stable'):
        if not neighbours[int(node)] & mask:
            mask |= 1 << int(node)
    return mask


def frozen_equivalence(seeds=(1017, 1019), graphs=3, rounds=6):
    """t11 acceptance (2): the tabu driver with no memory *is* the frozen search.

    ``platform_search_tabu(fifo=None)`` delegates to ``mis_v3_search.platform_search``,
    so this is a regression test on the delegation: any divergence in the frozen
    control flow shows up as a different work/best size or round count.
    """
    from mis_v11_tabu import platform_search_tabu
    from mis_v3_search import SearchGraph, platform_search
    detail = []
    for graph_index in range(graphs):
        graph, adjacency, edge_index = build_cpu_graph(nodes=40, degree=5, seed=graph_index)
        nodes = graph.num_nodes
        start = greedy_mis(nodes, edge_index)
        for seed in seeds:
            topology = SearchGraph(nodes, edge_index, seed=seed)
            frozen = platform_search(topology, start, outer_rounds=rounds,
                                     deadline=float('inf'))
            variant = platform_search_tabu(topology, start, fifo=None,
                                           outer_rounds=rounds, deadline=float('inf'))
            same = (frozen.work == variant.work and frozen.best == variant.best
                    and frozen.rule_size == variant.rule_size
                    and frozen.exchanges == variant.exchanges
                    and frozen.neutral_decisions == variant.neutral_decisions
                    and frozen.outer_steps == variant.outer_steps)
            detail.append(dict(graph=graph_index, seed=seed, same=bool(same),
                               frozen=[frozen.work.bit_count(), frozen.best.bit_count()],
                               variant=[variant.work.bit_count(), variant.best.bit_count()]))
    return all(item['same'] for item in detail), detail


# --------------------------------------------------------------------------- #
# mode 1: CPU selftest
# --------------------------------------------------------------------------- #
def selftest(args):
    problem, source = reference_problem()
    graph, adjacency, edge_index = build_cpu_graph()
    solver = StubSolver(problem, graph.num_nodes)
    torch.manual_seed(11)
    x0 = torch.randint(0, 2, (graph.num_nodes, solver.num_k), dtype=solver.dtype)
    checks = {}

    gen = V11.generator_for('cpu', 1234)
    out = V11.unroll_objective(solver, graph, adjacency, None, x0, steps=solver.args.unroll_steps,
                               decode_every=solver.args.unroll_decode_every, generator=gen)
    loss = out['loss']
    grad, = torch.autograd.grad(loss, out['logits'], allow_unused=False)
    checks['finite'] = bool(torch.isfinite(loss) and torch.isfinite(out['logits']).all()
                            and torch.isfinite(grad).all())
    checks['nonzero_grad'] = bool(grad.abs().sum() > 0)
    # A.3 is a full local search (it may also *remove* a node), so the assertions
    # are: the decoded set is feasible, and my wrapper agrees with the reference
    # decoder bit for bit.
    decoded = problem.decode_gradient_flip(out['state'].detach(), adjacency, graph.b,
                                           return_solutions=True)
    wrapped, _ = V11.decode_size(problem, out['state'].detach(), adjacency, graph.b)
    checks['feasible'] = bool(independent(decoded.bool(), edge_index))
    checks['decode_matches_reference'] = bool(torch.equal(wrapped, decoded.sum(0)))
    feasible_state = decoded.detach()
    checks['decode_from_feasible_stays_feasible'] = bool(
        independent(problem.decode_gradient_flip(feasible_state, adjacency, graph.b,
                                                 return_solutions=True).bool(), edge_index))
    archive = V11.unroll(solver, graph, adjacency, None, x0, steps=20, decode_every=5,
                         generator=V11.generator_for('cpu', 99), archive=True, keep_grad=False)
    archive_sizes = out['sizes']
    checks['archive_reported'] = bool(archive['archive_size'] is not None
                                      and float(archive['archive_size'].max()) >= archive_sizes[-1])

    gen_a = V11.generator_for('cpu', 7)
    gen_b = V11.generator_for('cpu', 7)
    run_a = V11.unroll(solver, graph, adjacency, None, x0, steps=12, decode_every=4, generator=gen_a)
    run_b = V11.unroll(solver, graph, adjacency, None, x0, steps=12, decode_every=4, generator=gen_b)
    checks['same_seed_bitwise'] = bool(torch.equal(run_a['state'], run_b['state']))

    # adaptive-gate degrade path: force the clock gate to trip
    saved = V11.TIME_GATE_SECONDS
    V11.TIME_GATE_SECONDS = -1.0
    try:
        solver2 = StubSolver(problem, graph.num_nodes, steps=200)
        V11.unroll_graph_loss(solver2, graph, adjacency, None, x0)
        first = dict(solver2.v11_unroll)
        V11.unroll_graph_loss(solver2, graph, adjacency, None, x0)
        second = dict(solver2.v11_unroll)
        checks['degrade_recorded'] = bool(
            first['measured'][0]['steps'] == 200 and first['steps'] == V11.FALLBACK_STEPS
            and first['degraded_reason'] and second['measured'][-1]['steps'] == V11.FALLBACK_STEPS)
        checks['degrade_trace'] = bool(len(second['measured']) == 2
                                       and second['updates'] == 2
                                       and len(second['size_trace']) == 2)
    finally:
        V11.TIME_GATE_SECONDS = saved

    # 10. no-memory tabu driver == frozen search, bit for bit (t11 acceptance (2))
    equal, frozen_detail = frozen_equivalence()
    checks['frozen_equivalence'] = bool(equal)

    report = dict(mode='selftest', reference=source, nodes=graph.num_nodes,
                  num_k=solver.num_k, checks=checks, passed=all(checks.values()),
                  loss=float(loss), grad_norm=float(grad.norm()),
                  sizes=out['sizes'], frozen_equivalence=frozen_detail)
    return report


# --------------------------------------------------------------------------- #
# mode 2: engine gates on GPU
# --------------------------------------------------------------------------- #
def engine_gates(args):
    from eval_v2 import dataset_at, load_policy, sync
    from eval_v8_padded import shape_of
    from mis_v1 import adjacency
    from mis_v8_pad import PaddedSearch

    args.save_dir = args.save_dir
    solver = load_policy(args)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.graphs)
    n_max, capacity = shape_of(dataset, len(dataset))
    engine = PaddedSearch(solver, n_max=n_max, capacity=capacity, chunk=args.chunk,
                          decode_chunk=args.decode_chunk, dense=args.dense)
    problem = solver.problem
    report = dict(mode='engine', graphs=len(dataset), chunk=args.chunk, dense=bool(args.dense),
                  dtype=str(solver.dtype), n_max=n_max, capacity=capacity,
                  rounds=args.rounds, rows=[])

    for index in range(len(dataset)):
        graph = dataset[index].to(device)
        edges = graph.edge_index
        adj = adjacency(graph)
        nodes = int(graph.num_nodes)
        seed = args.seed + index
        engine.load(graph, edges, seed)
        state_after_load = engine.state.clone()
        noise_state = engine.generator.get_state()
        sync(device)

        # G-B(i): captured engine vs un-captured eager path, same seed, same steps
        eager_state, eager_best, eager_energy = engine.eager_step(args.rounds)
        masks = []
        steps, decode_seconds, blocks = engine.run(1e18, lambda s: masks.append(engine.decode_best(s)),
                                                  max_blocks=int(np.ceil(args.rounds / args.chunk)))
        captured_state = engine.state.clone()
        bitwise = bool(torch.equal(eager_state, captured_state)
                       and torch.equal(eager_best, engine.best_states)
                       and torch.equal(eager_energy, engine.best_energy))

        # G-B(ii): the training-side stepper reproduces one engine block bit for bit
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        gen.set_state(noise_state)
        engine.state.copy_(state_after_load)
        engine.best_states.copy_(state_after_load)
        engine.best_energy.copy_(problem.energy_func(engine.adjacency, engine.bias,
                                                     engine.state)[0])
        # the engine's own generator has been advanced by G-B(i); rewind it to the
        # post-load state so this block and ``unroll(steps=1)`` consume the same noise
        engine.generator.set_state(noise_state)
        # ``eager_step`` restores its buffers and *returns* the stepped state
        engine_state, _, _ = engine.eager_step(1)
        # Compare on the *padded* layout: the engine runs the policy on the (n_max, K)
        # buffer through ``dense_norm``.  A per-graph (num_nodes, K) rollout uses a
        # different fp16 kernel shape, so cross-shape bit identity is not defined;
        # feeding the same buffer/dense matrix is what makes "bit for bit" meaningful.
        weight = engine.dense_norm if engine.use_dense else engine.normalized
        padded = SimpleNamespace(num_nodes=n_max, b=engine.bias, edge_index=edges,
                                 idx=index)
        gen.set_state(noise_state)
        roll = V11.unroll(solver, padded, weight, weight, state_after_load.clone(),
                          steps=1, decode_every=10 ** 6, generator=gen, keep_grad=False)
        stepper_bitwise = bool(torch.equal(roll['state'], engine_state))

        # G-B(iii): captured decoder vs reference A.3 on the decoded size
        engine_mask = engine.decode_best(engine.best_states)
        bits = unpack(engine_mask, nodes)
        decoded = problem.decode_gradient_flip(engine.best_states[:nodes].clone(), adj, graph.b,
                                              return_solutions=True)
        reference_size = int(decoded.sum(0).max())
        feasible_engine = bool(independent(bits, graph.edge_index.cpu()))

        # G-C: deadline semantics -- short budget, one block of overshoot at most
        engine.load(graph, edges, seed)
        sync(device)
        budget = args.budget
        tick = perf_counter()
        deadline = tick + budget
        short_masks = []
        steps_short, _, blocks_short = engine.run(deadline, lambda s: short_masks.append(engine.decode_best(s)))
        sync(device)
        wall = perf_counter() - tick
        ms_per_step = 1e3 * wall / max(steps_short, 1)
        block_time = wall / max(blocks_short, 1)
        deadline_ok = bool(short_masks and steps_short > 0
                           and wall <= budget + 1.5 * block_time)

        # fixed-step semantics (what --fixed_steps relies on)
        engine.load(graph, edges, seed)
        fixed_blocks = int(np.ceil(FIXED_STEPS / args.chunk))
        fixed_masks = []
        steps_fixed, _, blocks_fixed = engine.run(1e18, lambda s: fixed_masks.append(engine.decode_best(s)),
                                                 max_blocks=fixed_blocks)
        fixed_ok = bool(steps_fixed == fixed_blocks * args.chunk and blocks_fixed == fixed_blocks)

        # G-D: same seed -> identical masks and states
        engine.load(graph, edges, seed)
        repeat_masks = []
        engine.run(1e18, lambda s: repeat_masks.append(engine.decode_best(s)), max_blocks=fixed_blocks)
        reproducible = bool(repeat_masks == fixed_masks)

        report['rows'].append(dict(
            graph_id=index, nodes=nodes, seed=seed,
            eager_vs_captured_bitwise=bitwise,
            unroll_vs_engine_step_bitwise=stepper_bitwise,
            engine_decode_size=int(engine_mask.bit_count()),
            reference_decode_size=reference_size,
            decode_size_matches=bool(int(engine_mask.bit_count()) == reference_size),
            engine_decode_feasible=feasible_engine,
            deadline_wall_s=round(wall, 4), deadline_budget_s=budget,
            deadline_steps=steps_short, deadline_blocks=blocks_short,
            deadline_ms_per_step=round(ms_per_step, 3),
            deadline_ok=deadline_ok,
            fixed_steps=steps_fixed, fixed_blocks=blocks_fixed, fixed_steps_ok=fixed_ok,
            same_seed_reproducible=reproducible,
            capture_seconds=round(float(getattr(engine, 'capture_s', 0.0)), 4),
        ))
        engine.release()

    rows = report['rows']
    report['checks'] = dict(
        eager_vs_captured_bitwise=all(row['eager_vs_captured_bitwise'] for row in rows),
        unroll_vs_engine_step_bitwise=all(row['unroll_vs_engine_step_bitwise'] for row in rows),
        decode_size_matches=all(row['decode_size_matches'] for row in rows),
        engine_decode_feasible=all(row['engine_decode_feasible'] for row in rows),
        deadline_ok=all(row['deadline_ok'] for row in rows),
        fixed_steps_ok=all(row['fixed_steps_ok'] for row in rows),
        same_seed_reproducible=all(row['same_seed_reproducible'] for row in rows),
    )
    report['passed'] = all(report['checks'].values())
    report['ms_per_step'] = round(float(np.mean([row['deadline_ms_per_step'] for row in rows])), 3)
    return report


# --------------------------------------------------------------------------- #
# mode 3: gate 0 (A0 dual caliber)
# --------------------------------------------------------------------------- #
def read_artifact(path, budget=1.0):
    """dtype / dense / fp32_csr / fixed_steps of the arm that *ran* (t11 / R2).

    Read from the ``--v11_csv`` rows of the tested cell, aggregated over every row:
    a gate may pass only when all rows agree (100 % consistency), so a run that
    mixed configurations cannot pass by picking one lucky row.  Reading the
    artefact instead of the CLI flags is the point of the fix -- ``--expect_dtype
    fp16`` compared the request with itself and could never fail.
    """
    rows = []
    with Path(path).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if abs(float(row.get('budget') or 0.0) - budget) > 1e-9:
                continue
            if 'captured' not in (row.get('cell') or ''):
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f'no padded/captured rows at budget {budget} in {path}')
    values = {}
    for column in ('dtype', 'dense', 'fp32_csr', 'fixed_steps', 'steps'):
        values[column] = sorted({str(row.get(column, '')).strip() for row in rows})
    single = {column: (seen[0] if len(seen) == 1 else None)
              for column, seen in values.items()}
    return dict(path=str(path), rows=len(rows), values=values,
                consistent=all(len(seen) == 1 for seen in values.values()),
                dtype=single['dtype'], dense=single['dense'],
                fp32_csr=single['fp32_csr'], fixed_steps=single['fixed_steps'])


def gate0(args):
    rows = []
    with Path(args.csv).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            if abs(float(row['budget']) - 1.0) > 1e-9:
                continue
            if 'captured' not in row['cell']:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f'no padded/captured rows at budget 1.0 in {args.csv}')
    # One row per (graph, repeat): average repeats first so every graph weighs the same
    # as in the reference CSV, and count *graphs* (not rows) for the figure check.
    by_graph = {}
    for row in rows:
        by_graph.setdefault(row['graph_id'], []).append(row)
    sizes = [statistics.fmean(float(r['final_size']) for r in group) for group in by_graph.values()]
    search = [statistics.fmean(float(r['search_size']) for r in group) for group in by_graph.values()]
    steps = [float(row['steps']) for row in rows]
    wall = [float(row.get('solve_time_s') or 0.0) * 1e3 for row in rows]
    ms = [float(row.get('ms_per_step') or 0.0) for row in rows]
    mean_size = statistics.fmean(sizes)
    mean_steps = statistics.fmean(steps)
    mean_ms = statistics.fmean([value for value in ms if value]) if any(ms) else float('nan')
    summary = json.loads(Path(args.json).read_text(encoding='utf-8')) if args.json else {}
    mean_search = statistics.fmean(search)
    if not getattr(args, 'v11_csv', None):
        raise SystemExit('gate0 needs --v11_csv: dtype/dense/fp32_csr must be read from '
                         'the tested run, not from the expectation flags (t11 / R2)')
    artifact = read_artifact(args.v11_csv)
    checks = dict(
        figures=len(by_graph) == args.expect_graphs,
        # search_size is the only judgement column (R3): it excludes the PAVE reserve
        search=abs(mean_search - REFERENCE_SEARCH) <= args.tolerance,
        steps_810=all(float(row['steps']) == FIXED_STEPS_RUN for row in rows),
        dtype_fp16=artifact['dtype'] == 'fp16',
        dense=artifact['dense'] in ('1', 'True', 'true'),
        fixed_steps=artifact['fixed_steps'] in (str(FIXED_STEPS), f'{FIXED_STEPS}.0'),
        not_fp32_csr=(artifact['fp32_csr'] in ('0', '0.0')
                      and not (mean_ms == mean_ms
                               and FP32_CSR_MS * 0.8 <= mean_ms <= FP32_CSR_MS * 1.2)),
        artifact_consistent=bool(artifact['consistent']),
    )
    report = dict(mode='gate0', csv=str(args.csv), json=str(args.json) if args.json else None,
                  v11_csv=str(args.v11_csv),
                  graphs=len(by_graph), rows=len(rows), tolerance=args.tolerance,
                  reference=REFERENCE_SEARCH,
                  reference_kind='A0_declared_config_fp16_dense_810_steps',
                  mean_search_size=round(mean_search, 4),
                  delta_search=round(mean_search - REFERENCE_SEARCH, 4),
                  config_sensitivity=CONFIG_SENSITIVITY,
                  config_sensitivity_note='fp32<->fp16, deterministic, not in the tolerance',
                  historical_clock_reference=HISTORICAL_CLOCK_REFERENCE,
                  historical_note='historical_clock_reference_unreproducible',
                  delta_vs_historical=round(mean_search - HISTORICAL_CLOCK_REFERENCE, 4),
                  # final_size is reported, never judged: the PAVE reserve makes it move
                  mean_final_size=round(mean_size, 4),
                  final_size_note=f'report only; contains PAVE 0.2 s reserve jitter '
                                  f'+-{FINAL_SIZE_JITTER}, never use for <=0.001 decisions',
                  mean_steps=round(mean_steps, 1), mean_ms_per_step=round(mean_ms, 3),
                  artifact=artifact, checks=checks, passed=all(checks.values()),
                  summary_keys=sorted(summary)[:20])
    return report


# --------------------------------------------------------------------------- #
# gate 0 must be able to fail (t11 acceptance (1))
# --------------------------------------------------------------------------- #
NEGATIVE_FIELDS = dict(
    legacy=['graph_id', 'budget', 'cell', 'search_size', 'final_size', 'steps',
            'ms_per_step', 'solve_time_s', 'seed', 'repeat', 'arm', 'decoder'],
    v11=['graph_id', 'budget', 'cell', 'arm', 'decoder', 'seed', 'repeat',
         'search_size', 'final_size', 'steps', 'wall_ms', 'ms_per_step', 'search_ms',
         'load_ms', 'blocks', 'pave_gain', 'deadline_overrun', 'temperature',
         'anneal', 'fixed_steps', 'dtype', 'dense', 'fp32_csr'],
)


def synthetic_rows(graphs=3, repeats=2):
    legacy, v11 = [], []
    for index in range(graphs):
        for repeat in range(repeats):
            row = dict(graph_id=f'graph_{index:06d}', budget=1.0,
                       cell='padded/captured', search_size=REFERENCE_SEARCH,
                       final_size=REFERENCE_SEARCH + 0.0005, steps=FIXED_STEPS_RUN,
                       ms_per_step=1.28, solve_time_s=1.04, seed=1017, repeat=repeat,
                       arm='padded', decoder='captured')
            legacy.append(row)
            v11.append(dict(row, wall_ms=1040.0, search_ms=1010.0, load_ms=6.7,
                            blocks=FIXED_STEPS_RUN // 10, pave_gain=0,
                            deadline_overrun=1, temperature=1.0, anneal=0,
                            fixed_steps=FIXED_STEPS, dtype='fp16', dense=1,
                            fp32_csr=0))
    return legacy, v11


def write_rows(path, fields, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def negative_selftest(args):
    """A gate that cannot fail is not a gate: corrupt the artefact and require exit != 0."""
    import tempfile
    legacy, v11 = synthetic_rows()
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)

        def run(name, legacy_rows, v11_rows):
            legacy_path, v11_path = folder / f'{name}.csv', folder / f'{name}_v11.csv'
            write_rows(legacy_path, NEGATIVE_FIELDS['legacy'], legacy_rows)
            write_rows(v11_path, NEGATIVE_FIELDS['v11'], v11_rows)
            return gate0(SimpleNamespace(csv=legacy_path, v11_csv=v11_path, json=None,
                                         expect_graphs=3, tolerance=REFERENCE_SEARCH_TOLERANCE))

        clean = run('clean', legacy, v11)
        shifted = [dict(row, search_size=row['search_size'] + 0.05) for row in legacy]
        shift = run('shift', shifted, v11)
        fp32 = [dict(row, dtype='fp32', dense=0, fp32_csr=1) for row in v11]
        config = run('fp32', legacy, fp32)
    checks = dict(clean_passes=bool(clean['passed']),
                  shift_fails=not bool(shift['passed']) and shift['checks']['search'] is False,
                  config_fails=(not bool(config['passed'])
                                and config['checks']['dtype_fp16'] is False
                                and config['checks']['dense'] is False
                                and config['checks']['not_fp32_csr'] is False))
    return dict(mode='gate0-negative', checks=checks, passed=all(checks.values()),
                clean=clean['checks'], shift=shift['checks'], fp32=config['checks'])


# --------------------------------------------------------------------------- #
# mode 4: A1 vs A1' audit (direction §2.5: "only the loss may differ")
# --------------------------------------------------------------------------- #
def audit(args):
    left = json.loads(Path(args.a).read_text(encoding='utf-8'))
    right = json.loads(Path(args.b).read_text(encoding='utf-8'))
    left_args = left.get('argv', {})
    right_args = right.get('argv', {})
    ignore = {'loss', 'run_json', 'save_dir', 'log_file'}
    keys = sorted(set(left_args) | set(right_args))
    diffs = {key: [left_args.get(key), right_args.get(key)] for key in keys
             if key not in ignore and left_args.get(key) != right_args.get(key)}
    checks = dict(only_loss_differs=not diffs,
                  loss_differs=left.get('loss') != right.get('loss'),
                  same_unroll=left.get('unroll', {}).get('steps')
                  == right.get('unroll', {}).get('steps'))
    report = dict(mode='audit', a=str(args.a), b=str(args.b),
                  losses=[left.get('loss'), right.get('loss')],
                  differing_args=diffs, checks=checks, passed=all(checks.values()))
    return report


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='v11 hard gates')
    parser.add_argument('mode', choices=('selftest', 'engine', 'gate0', 'audit'))
    parser.add_argument('--a', type=Path, help='audit: A1 run.json')
    parser.add_argument('--b', type=Path, help="audit: A1' run.json")
    parser.add_argument('--out', type=Path, default=Path('outputs/v11'))
    parser.add_argument('--csv', type=Path, help='gate0: eval_v8_padded rows csv')
    parser.add_argument('--json', type=Path, help='gate0: eval_v8_padded summary json')
    parser.add_argument('--v11_csv', type=Path,
                        help='gate0: per-graph v11 CSV; dtype/dense/fp32_csr/fixed_steps '
                             'are read from the arm that ran (t11 / R2)')
    parser.add_argument('--negative-selftest', action='store_true',
                        help='gate0: corrupt the artefact and require the gate to fail')
    parser.add_argument('--expect_graphs', type=int, default=1000)
    parser.add_argument('--tolerance', type=float, default=REFERENCE_SEARCH_TOLERANCE,
                        help='gate0: absolute tolerance on mean search_size. A0 is the '
                             'declared config (fp16+dense) at 810 fixed steps (%.4f); '
                             'fp32<->fp16 differs by %.3f (deterministic, not noise).'
                             % (REFERENCE_SEARCH, CONFIG_SENSITIVITY))
    parser.add_argument('--graphs', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=10)
    parser.add_argument('--decode_chunk', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=40)
    parser.add_argument('--budget', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1017)
    parser.add_argument('--dense', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='outputs/v2/none_seed0/checkpoints')
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--split', default='test')
    params = parser.parse_args()

    if params.mode == 'selftest':
        report = selftest(params)
    elif params.mode == 'engine':
        from eval_v2 import policy_parser
        # ``eval_v2.policy_parser`` marks --result_dir as required; the gates never write
        # eval rows, so point it at the gate output directory.
        policy = policy_parser().parse_args(['--result_dir', str(params.out)])
        policy.save_dir = params.save_dir
        policy.test_path = params.test_path
        policy.split = params.split
        policy.seed = params.seed
        report = engine_gates(SimpleNamespace(**{**vars(policy), **vars(params)}))
    elif params.mode == 'gate0':
        report = negative_selftest(params) if params.negative_selftest else gate0(params)
    else:
        report = audit(params)

    params.out.mkdir(parents=True, exist_ok=True)
    suffix = '_negative' if (params.mode == 'gate0' and params.negative_selftest) else ''
    path = params.out / f'gates_{params.mode}{suffix}.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + '\n',
                    encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('mode', 'checks', 'passed')}, indent=2,
                     ensure_ascii=False))
    print(f'wrote {path}')
    if not report['passed']:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
