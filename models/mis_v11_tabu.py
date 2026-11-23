"""v11 C4/L04: node-identity FIFO tabu on the plateau search (new file, no frozen edits).

Direction lock `docs/temporary/RLNN_v11_Direction_Locked.md` §3 wants the *action*
tabu of L04 (AlphaZero+Tabu, IJCAI 2024, PDF p.5 §5 + Alg.1) instead of the
node-level "forbidden" of v10.

Two corrections over the first v11 draft (both from the captain's code review of
2026-11-17 14:40, m01061):

* the draft keyed the FIFO on ``tuple(action.removed)`` and rejected a candidate
  only when an *identical removed set* had just been executed.  Exchange actions
  almost never repeat a removed set, so the FIFO could not fire: ER was 0 for
  every depth h and the six h rows were bit-identical.  L04's tabu list lives on
  **node identity** (a node just evicted may not be re-added within h steps, a
  node just added may not be evicted within h steps), not on action identity;
* the measured quantity must therefore be defined with the node-level
  intersection, and the probe has to report how often the intersection was
  non-empty (``hits``) next to the pool sizes.

``TabuList`` implements that list.  Two intersection conventions are provided so
the probe can report both:

``cross`` (default, L04 Alg.1)
    forbid when ``set(action.added) & recently_removed`` or
    ``set(action.removed) & recently_added`` is non-empty.  This is the classic
    move-tabu pair: the *undo* of the last h exchanges is forbidden, which is
    precisely the "do not walk straight back" rule the direction lock asks for.
``same``
    forbid when ``set(action.added) & recently_added`` or
    ``set(action.removed) & recently_removed`` is non-empty (the literal reading
    of the review comment).  Kept for completeness; on a walk that alternates
    add/remove of different nodes it fires much less often.

Metric definitions (also written into the probe JSON, direction lock §3.4):

``ER`` elimination rate
    ``ER = 1 - |pool_tabu| / |pool_plain|``, evaluated **paired** on the same
    ``(graph, state, outer round)``.  Denominator = number of macro rounds in
    which a candidate pool was actually built; each round contributes its own
    ratio (``er_rounds`` says how many).  ``er_pooled = 1 -
    sum(pool_tabu)/sum(pool_plain)`` is reported next to it so a single huge pool
    cannot dominate.
``AR`` acceptance rate
    ``AR = (rounds with size_after == size_before and mask changed and archive
    NOT refreshed) / outer_rounds``.  The "not refreshed" clause is what makes
    AR measure *walking* rather than *improving*: a round that improves the
    archive is not a plateau walk.  ``refreshed_rounds`` is reported separately
    because "no refresh in 8 rounds" is the second half of the C4 sentence.

**AR is a mechanism metric only** (t11 §3.6 closure): ``AR ~ 0.99`` says "the walk
happens" -- it is the same value with and without tabu -- and must never be read
as "exploration is sufficient" or as a gain.  The only gain metrics are ``gain``
and ``refreshed_rounds``.

Why a separate module: ``models/mis_v3_search.py`` is on the frozen list, and the
existing code has two structural gaps that this module fills without touching it

* ``models/mis_v3_search.py:197 candidate_actions`` forbids only the exact reverse
  of the actions in ``tabu`` (depth 1) -- "do not undo immediately", not "do not
  repeat recently";
* ``models/mis_v3_search.py:247 repair()`` has no ``tabu`` parameter at all, so
  every destroy-repair converges back to the same local optimum (the code-level
  cause of the v10 "same solution in 20/20 graphs" result).

Everything here is a thin wrapper/re-implementation over the frozen primitives
(``graph._exchanges``, ``graph.apply``, ``candidate_actions``, ``rollout``), so a
probe can measure the walk without changing the deployment path.
"""
from collections import deque
from math import inf
from time import perf_counter

from mis_v3_search import CandidatePool, RepairResult, SearchResult, candidate_actions, choose_action


class _Entry:
    """Minimal action record: the FIFO only needs the two node tuples."""

    __slots__ = ("added", "removed")

    def __init__(self, added=(), removed=()):
        self.added = tuple(added)
        self.removed = tuple(removed)

    def __repr__(self):
        return f"_Entry(added={self.added}, removed={self.removed})"


class TabuList:
    """Node-identity FIFO of depth ``h`` over executed exchange actions (L04 Alg.1)."""

    def __init__(self, h, mode="cross"):
        if mode not in ("cross", "same"):
            raise ValueError(f"unknown tabu mode {mode!r}")
        self.h = int(h)
        self.mode = mode
        self.fifo = deque(maxlen=self.h) if self.h else None
        self.added = set()
        self.removed = set()
        self.checks = 0
        self.hits = 0
        self.sample = None

    # -- state ---------------------------------------------------------------
    def __bool__(self):
        return bool(self.fifo)

    def __len__(self):
        return len(self.fifo) if self.fifo is not None else 0

    def clear(self):
        if self.fifo is not None:
            self.fifo.clear()
        self.added.clear()
        self.removed.clear()

    def push(self, action):
        """Record an executed action; the sets are rebuilt so eviction is exact."""
        if self.fifo is None:
            return
        self.fifo.append((tuple(action.added), tuple(action.removed)))
        self.added = {node for added, _ in self.fifo for node in added}
        self.removed = {node for _, removed in self.fifo for node in removed}

    # -- query ---------------------------------------------------------------
    def forbids(self, action):
        if self.fifo is None or not self.fifo:
            return False
        self.checks += 1
        added = set(action.added)
        removed = set(action.removed)
        if self.mode == "cross":
            hit = (added & self.removed) or (removed & self.added)
        else:
            hit = (added & self.added) or (removed & self.removed)
        if hit:
            self.hits += 1
            if self.sample is None:
                self.sample = dict(mode=self.mode,
                                   action_added=sorted(added), action_removed=sorted(removed),
                                   tabu_added=sorted(self.added), tabu_removed=sorted(self.removed),
                                   intersect=sorted(hit))
            return True
        return False

    def stats(self):
        return dict(h=self.h, mode=self.mode, depth=len(self), checks=self.checks,
                    hits=self.hits, sample=self.sample)


def improving_actions_tabu(graph, selected, *, cap=48, anchor=None, deadline=inf, tabu=None):
    """``improving_actions`` minus the actions the node-identity FIFO forbids.

    Mirrors the frozen rule (prefer size-1 exchanges; fall back to size-2 only when
    the size-1 list is empty) but filters *before* the cap, otherwise tabu would be
    applied to an already-truncated pool and the measured rejection rate would be
    meaningless.
    """
    for size in (1, 2):
        found = []
        for action in graph._exchanges(selected, size, deadline):
            if anchor is not None and anchor not in action.removed:
                continue
            if tabu is not None and tabu.forbids(action):
                continue
            found.append(action)
            if len(found) >= cap:
                break
        if found or size == 2:
            return found
    return []


def repair_tabu(graph, selected, *, tabu=None, rounds=4, cap=48, anchor=None,
                deadline=inf, extend_first=True):
    """``models/mis_v3_search.py:247 repair()`` with the node-identity FIFO.

    The executed action of every inner round is pushed onto ``tabu`` (shared across
    outer rounds by the caller), which is what the frozen ``repair()`` lacks.
    """
    result = RepairResult(selected)
    if perf_counter() >= deadline:
        result.deadline_hit = True
        return result
    if extend_first:
        candidate = graph.extend(selected)
        if perf_counter() >= deadline:
            result.deadline_hit = True
            return result
        result.work = candidate
    for _ in range(rounds):
        actions = improving_actions_tabu(graph, result.work, cap=cap, anchor=anchor,
                                         deadline=deadline, tabu=tabu)
        result.candidate_counts.append(len(actions))
        if perf_counter() >= deadline:
            result.deadline_hit = True
            return result
        if not actions:
            result.exhausted = True
            return result
        action = actions[0]
        candidate = graph.apply(result.work, action)
        if perf_counter() >= deadline:
            result.deadline_hit = True
            return result
        result.work = candidate
        result.exchanges += 1
        if tabu is not None:
            tabu.push(action)
    result.capped = True
    return result


def pool_plain_and_tabu(graph, selected, *, tabu=None, cap=32, anchored=False,
                        history=(), deadline=inf):
    """Candidate pool without and with the FIFO, paired on the same state.

    Returns ``(plain_pool, tabu_pool, rejected)`` where ``rejected`` counts the
    plain actions the FIFO removed (the numerator behind ``ER``).
    """
    plain = candidate_actions(graph, selected, cap=cap, anchored=anchored,
                             tabu=history, deadline=deadline)
    if tabu is None or not tabu:
        return plain, plain, 0
    kept = [action for action in plain.actions if not tabu.forbids(action)]
    return plain, CandidatePool(kept, len(kept), plain.completed), len(plain.actions) - len(kept)


def platform_search_tabu(graph, selected, *, tabu=None, fifo=None, choose="coverage",
                         outer_rounds=8, initial_rounds=3, horizon=2, candidate_cap=32,
                         improving_cap=48, repair_rounds=4, anchored=False, history=(),
                         deadline=inf, trailing=2, on_round=None):
    """``models/mis_v3_search.py:340 platform_search()`` with node-identity tabu.

    ``fifo`` is an alias of ``tabu`` (the reviewer's spelling).

    Passing ``tabu=None``/``fifo=None`` means "no memory": the call is delegated to
    the frozen ``mis_v3_search.platform_search`` unchanged, so the two are bit for
    bit identical (gates selftest item 10).  An *empty* ``TabuList`` instead takes
    the loop below, which additionally appends the executed action to ``history``
    (the probe's h=0 control keeps that bookkeeping).

    Differences from the frozen version once a FIFO is supplied, all additive:
    * the FIFO threads through ``repair_tabu``, the candidate pool and
      ``rollout_tabu``;
    * every outer round reports ``(size_before, size_after, changed, refreshed,
      pool_plain, pool_tabu, rejected)`` through ``on_round`` so the probe can
      compute ER and AR.  (No callback fires on the ``tabu=None`` delegation.)

    Deadline semantics are the frozen ones (``deadline=inf`` disables all clock
    reasoning); the probe runs with ``deadline=inf`` so that all 8 rounds happen
    and AR is measured on a fixed denominator.
    """
    if tabu is None:
        tabu = fifo
    elif fifo is not None:
        raise ValueError('pass either tabu= or fifo=, not both')
    if tabu is None:
        from mis_v3_search import platform_search as frozen_platform_search
        return frozen_platform_search(graph, selected, choose=choose,
                                      outer_rounds=outer_rounds,
                                      initial_rounds=initial_rounds, horizon=horizon,
                                      candidate_cap=candidate_cap,
                                      improving_cap=improving_cap,
                                      repair_rounds=repair_rounds, anchored=anchored,
                                      history=history, deadline=deadline)
    limited = deadline != inf
    work_deadline = deadline - 0.002 if limited else inf
    initial = (repair_tabu(graph, selected, tabu=tabu, rounds=initial_rounds,
                           cap=improving_cap, deadline=work_deadline, extend_first=False)
               if initial_rounds else RepairResult(selected))
    result = SearchResult(initial.work, initial.work, tuple(history)[-trailing:],
                          selected.bit_count(), initial.work.bit_count(),
                          exchanges=initial.exchanges, deadline_hit=initial.deadline_hit)
    for step in range(outer_rounds):
        if result.deadline_hit or (limited and perf_counter() >= work_deadline):
            result.deadline_hit = True
            break
        before = result.work
        began = perf_counter()
        strict = improving_actions_tabu(graph, result.work, cap=1, deadline=work_deadline,
                                        tabu=tabu)
        pool_plain = pool_tabu = rejected = None
        if strict:
            segment = repair_tabu(graph, before, tabu=tabu, rounds=repair_rounds,
                                  cap=improving_cap, deadline=work_deadline)
            result.work = segment.work
            result.exchanges += segment.exchanges
            result.deadline_hit = segment.deadline_hit
            kind = "repair"
        else:
            plain, pool, rejected = pool_plain_and_tabu(graph, before, tabu=tabu, cap=candidate_cap,
                                                        anchored=anchored, history=result.history,
                                                        deadline=work_deadline)
            pool_plain, pool_tabu = len(plain.actions), len(pool.actions)
            if not pool.completed:
                result.deadline_hit = True
                break
            if not pool.actions:
                break
            action = (choose(graph, before, pool.actions, (outer_rounds - step) / outer_rounds)
                      if callable(choose) else choose_action(graph, before, pool.actions, choose))
            result.history = (result.history + (action,))[-trailing:]
            macro = rollout_tabu(graph, before, action, tabu=tabu, horizon=horizon,
                                 repair_rounds=repair_rounds, improving_cap=improving_cap,
                                 candidate_cap=candidate_cap, deadline=work_deadline)
            result.work = macro.work
            result.exchanges += macro.exchanges
            result.neutral_decisions += len(macro.attempted_actions)
            result.deadline_hit = macro.deadline_hit
            kind = "macro"
        after = result.work
        refreshed = after.bit_count() > result.best.bit_count()
        if refreshed:
            result.best = after
        if on_round is not None:
            on_round(dict(round=step, kind=kind, size_before=before.bit_count(),
                          size_after=after.bit_count(), changed=after != before,
                          refreshed=refreshed, seconds=perf_counter() - began,
                          pool_plain=pool_plain, pool_tabu=pool_tabu, rejected=rejected))
        result.trace.append(dict(round=step, kind=kind, size=after.bit_count()))
        if result.deadline_hit:
            break
    result.outer_steps = len(result.trace)
    return result


def rollout_tabu(graph, selected, action, *, tabu=None, horizon=2, repair_rounds=4,
                 improving_cap=48, candidate_cap=32, deadline=inf):
    """``models/mis_v3_search.py:282 rollout()`` with the FIFO threaded through repair."""
    from mis_v3_search import RolloutResult

    result = RolloutResult(selected, selected, 0)
    attempted = [action]
    work = graph.apply_move(selected, action)
    anchor = action.added[0] if len(action.removed) == 2 else None
    for stage in range(horizon):
        if stage:
            strict = improving_actions_tabu(graph, work, cap=1, deadline=deadline, tabu=tabu)
            if not strict:
                _, pool, _ = pool_plain_and_tabu(graph, work, tabu=tabu, cap=candidate_cap,
                                                 history=(action,), deadline=deadline)
                if not pool.completed or not pool.actions:
                    break
                second = choose_action(graph, work, pool.actions)
                work = graph.apply_move(work, second)
                attempted.append(second)
        segment = repair_tabu(graph, work, tabu=tabu, rounds=repair_rounds,
                              cap=improving_cap, anchor=anchor if stage == 0 else None,
                              deadline=deadline)
        work = segment.work
        result.segments += 1
        result.exchanges += segment.exchanges
        result.capped |= segment.capped
        if work.bit_count() > result.best.bit_count():
            result.best = work
        if segment.deadline_hit:
            result.deadline_hit = True
            break
    result.ended_below_start = work.bit_count() < selected.bit_count()
    result.rolled_back = work.bit_count() < result.best.bit_count()
    result.work = result.best if result.rolled_back else work
    result.returned_to_start = result.work == selected
    result.attempted_actions = tuple(attempted)
    result.neutral_actions = () if result.rolled_back else tuple(attempted)
    result.gain = result.best.bit_count() - selected.bit_count()
    return result


# ---------------------------------------------------------------------------
# self test (captain's diagnostic (1): a construction that *must* fire tabu)
# ---------------------------------------------------------------------------
def self_test(verbose=True):
    """Assert the FIFO actually rejects candidates (``pool_tabu`` strict subset).

    Part A is synthetic and deterministic: the undo of an exchange must be
    forbidden in ``cross`` mode and allowed in ``same`` mode, which pins down the
    semantics the probe reports.

    Part B runs on a real RB graph: build the candidate pool of a real state, mark
    the "added" node of one pool action as recently evicted (that is exactly the
    situation after executing it), and require that the action disappears from the
    tabu pool while the plain pool still contains it.
    """
    undo = (_Entry(added=(7,), removed=(3,)), _Entry(added=(3,), removed=(7,)))
    executed, inverse = undo
    results = {}
    for mode in ("cross", "same"):
        tabu = TabuList(8, mode=mode)
        tabu.push(executed)
        results[mode] = tabu.forbids(inverse)
    if verbose:
        print(f"[A] undo forbidden: cross={results['cross']} same={results['same']}")
    if not results["cross"]:
        raise SystemExit("self_test: cross mode must forbid the undo of an exchange")

    import torch

    from mis_v3_search import SearchGraph

    # deterministic synthetic graph (no dataset / no torch_geometric needed): a
    # union of cycles, i.e. a 4-regular-ish sparse graph with a non-trivial MIS
    nodes = 120
    edges = []
    for offset in (1, 7, 31):
        for node in range(nodes):
            edges.append((node, (node + offset) % nodes))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    graphs = [SearchGraph(nodes, edge_index, seed=1017)]
    checked = fired = 0
    for index, graph in enumerate(graphs):
        start = 0
        for node in sorted(range(graph.num_nodes), key=lambda n: graph.neighbors[n].bit_count()):
            if not graph.neighbors[node] & start:
                start |= 1 << node
        plain, _, _ = pool_plain_and_tabu(graph, start, tabu=None, cap=32)
        if not plain.actions:
            continue
        action = plain.actions[0]
        tabu = TabuList(8, mode="cross")
        tabu.push(_Entry(added=(), removed=action.added))     # "that node was just evicted"
        plain2, pool2, rejected = pool_plain_and_tabu(graph, start, tabu=tabu, cap=32)
        checked += 1
        strict = len(pool2.actions) < len(plain2.actions)
        fired += int(strict and rejected > 0)
        if verbose:
            print(f"[B] graph {index}: |pool|={len(plain2.actions)} "
                  f"|pool_tabu|={len(pool2.actions)} rejected={rejected} strict={strict}")
        if not strict:
            raise SystemExit(f"self_test: FIFO did not shrink the pool on graph {index}")
    if not checked:
        raise SystemExit("self_test: no candidate pool was built")
    return dict(cross_forbids_undo=results["cross"], same_forbids_undo=results["same"],
                graphs_checked=checked, graphs_strict_subset=fired)


if __name__ == "__main__":
    print(self_test())
