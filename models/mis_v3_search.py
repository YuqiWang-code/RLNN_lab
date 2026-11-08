"""Finite plateau search and paired continuation values on legal MIS solutions."""
from dataclasses import dataclass, field
from itertools import chain, islice
from math import inf, isfinite, log1p
import random
from time import perf_counter

from mis_v2_exchange import Exchange, ExchangeGraph, node_mask, nodes


STATS_NAMES = (
    "removed_log_degree", "added_log_degree", "removed_count", "added_tightness",
    "removed_log_r1", "removed_log_r2", "added_log_r1", "added_log_r2",
    "log_release_count", "log_release_compatible_pairs", "log_covered",
    "log_shared_coverage", "added_log_outside_degree", "removed_log_outside_degree",
    "log_pairs_before", "log_pairs_after", "new_zero_tight", "lost_zero_tight",
    "new_one_tight", "lost_one_tight", "new_two_tight", "lost_two_tight",
    "selected_fraction", "remaining_fraction",
)
STATS_DIM = len(STATS_NAMES)


def reverse(action):
    return action.added, action.removed


@dataclass
class CandidatePool:
    actions: list
    total_count: int
    completed: bool = True

    @property
    def exact(self):
        return self.completed


@dataclass
class RepairResult:
    work: int
    exchanges: int = 0
    capped: bool = False
    exhausted: bool = False
    deadline_hit: bool = False
    candidate_counts: list = field(default_factory=list)


@dataclass
class RolloutResult:
    work: int
    best: int
    gain: int
    neutral_actions: tuple = ()
    attempted_actions: tuple = ()
    exchanges: int = 0
    segments: int = 0
    returned_to_start: bool = False
    ended_below_start: bool = False
    rolled_back: bool = False
    deadline_hit: bool = False
    capped: bool = False


@dataclass
class SearchResult:
    work: int
    best: int
    history: tuple
    base_size: int
    rule_size: int
    exchanges: int = 0
    neutral_decisions: int = 0
    outer_steps: int = 0
    deadline_hit: bool = False
    trace: list = field(default_factory=list)


class SearchGraph(ExchangeGraph):
    """Bitset topology with one fixed node priority for all candidate branches."""

    def __init__(self, num_nodes, edge_index, seed=0):
        super().__init__(num_nodes, edge_index)
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.seed = seed
        priority = list(range(num_nodes))
        random.Random(seed).shuffle(priority)
        self.priority = [0] * num_nodes
        for rank, node in enumerate(priority):
            self.priority[node] = rank

    def compatible_pairs(self, mask):
        count = 0
        while mask:
            bit = mask & -mask
            mask ^= bit
            count += (mask & ~self.neighbors[bit.bit_length() - 1]).bit_count()
        return count

    def state_structure(self, selected):
        tightness = [(adj & selected).bit_count() for adj in self.neighbors]
        masks = [0, 0, 0]
        for node in nodes(self.all_nodes ^ selected):
            count = tightness[node]
            if count < 3:
                masks[count] |= 1 << node
        r1 = [(adj & masks[1]).bit_count() for adj in self.neighbors]
        r2 = [(adj & masks[2]).bit_count() for adj in self.neighbors]
        pairs = [self.compatible_pairs(self.neighbors[i] & masks[1])
                 if selected & (1 << i) else 0 for i in range(self.num_nodes)]
        return tightness, masks, r1, r2, pairs

    def node_features(self, selected):
        tightness, _, r1, r2, pairs = self.state_structure(selected)
        return [[float(bool(selected & (1 << i))), min(tightness[i], 3) / 3,
                 self.log_degree[i], float(not selected & (1 << i) and tightness[i] == 1),
                 float(not selected & (1 << i) and tightness[i] == 2),
                 log1p(r1[i]), log1p(r2[i]), log1p(pairs[i])]
                for i in range(self.num_nodes)]

    def affected_nodes(self, action):
        affected = 0
        for node in action.removed + action.added:
            affected |= self.neighbors[node]
        return list(nodes(affected))

    def priority_key(self, action):
        return tuple(self.priority[i] for i in action.added + action.removed)

    def heuristic_key(self, selected, action):
        return (*self.heuristic(selected, action), self.priority_key(action))

    def apply_move(self, selected, action):
        candidate = (selected & ~node_mask(action.removed)) | node_mask(action.added)
        if not self.feasible(candidate):
            raise ValueError("Move must preserve independence")
        return candidate

    def stats_features(self, selected, actions, remaining=1.0):
        tight, masks, r1, r2, pairs = self.state_structure(selected)
        outside = self.all_nodes ^ selected
        levels = [0] * (3 + max((len(action.removed) for action in actions), default=0))
        for node, count in enumerate(tight):
            if count < len(levels):
                levels[count] |= 1 << node
        log_pairs_before = log1p(sum(pairs))
        selected_fraction = selected.bit_count() / self.num_nodes
        rows = []
        for action in actions:
            successor = self.apply_move(selected, action)
            removed = node_mask(action.removed)
            removed_neighbors = 0
            after_levels = levels
            for node in action.removed:
                adjacent = self.neighbors[node]
                removed_neighbors |= adjacent
                after_levels = [(same & ~adjacent) | (higher & adjacent)
                                for same, higher in zip(after_levels, after_levels[1:])]
            after_levels = after_levels[:3]
            added_neighbors = 0
            for node in action.added:
                adjacent = self.neighbors[node]
                added_neighbors |= adjacent
                zero, one, two = after_levels
                after_levels = [zero & ~adjacent,
                                (one & ~adjacent) | (zero & adjacent),
                                (two & ~adjacent) | (one & adjacent)]
            after_masks = [mask & ~successor for mask in after_levels]
            after_pairs = sum(self.compatible_pairs(self.neighbors[node] & after_masks[1])
                              for node in nodes(successor))
            released = removed_neighbors & outside
            for node in nodes(selected & ~removed):
                released &= ~self.neighbors[node]
            removed_cover = removed_neighbors & outside
            added_cover = added_neighbors & outside & ~node_mask(action.added)
            row = [sum(self.log_degree[i] for i in action.removed),
                   sum(self.log_degree[i] for i in action.added), len(action.removed),
                   sum(tight[i] for i in action.added),
                   sum(log1p(r1[i]) for i in action.removed),
                   sum(log1p(r2[i]) for i in action.removed),
                   sum(log1p(r1[i]) for i in action.added),
                   sum(log1p(r2[i]) for i in action.added),
                   log1p(released.bit_count()), log1p(self.compatible_pairs(released)),
                   log1p(added_cover.bit_count()), log1p((removed_cover & added_cover).bit_count()),
                   sum(log1p((self.neighbors[i] & outside).bit_count()) for i in action.added),
                   sum(log1p((self.neighbors[i] & outside).bit_count()) for i in action.removed),
                   log_pairs_before, log1p(after_pairs)]
            for before, after in zip(masks, after_masks):
                row.extend(((after & ~before).bit_count(), (before & ~after).bit_count()))
            row.extend((selected_fraction, float(remaining)))
            rows.append(row)
        return rows


def candidate_actions(graph, selected, *, cap=32, anchored=False, tabu=(), deadline=inf):
    """Root history contains executed moves; only their exact reverse is excluded."""
    forbidden = {reverse(action) for action in tabu}
    outside = graph.all_nodes ^ selected
    blockers = [(node, graph.neighbors[node] & selected) for node in nodes(outside)]
    actions = []
    complete = True
    for node, blocked in blockers:
        if perf_counter() >= deadline:
            complete = False
            break
        count = blocked.bit_count()
        if count != 1 and not (anchored and count == 2):
            continue
        removed = tuple(nodes(blocked))
        if (removed, (node,)) in forbidden:
            continue
        released = sum(bool(mask and not mask & ~blocked) for _, mask in blockers)
        actions.append(Exchange(removed, (node,), released))
    total = len(actions)
    if cap is not None and total > cap:
        ordered = sorted(actions, key=lambda action: graph.heuristic_key(selected, action))
        first = (cap + 1) // 2
        actions = ordered[:first] + sorted(ordered[first:], key=graph.priority_key)[:cap - first]
    return CandidatePool(actions, total, complete)


def improving_actions(graph, selected, *, cap=48, union=False, anchor=None, deadline=inf):
    def allowed(size):
        for action in graph._exchanges(selected, size, deadline):
            if anchor is None or anchor not in action.removed:
                yield action

    if union:
        return list(islice(chain(allowed(1), allowed(2)), cap))
    actions = list(islice(allowed(1), cap))
    return actions if actions else list(islice(allowed(2), cap))


def choose_action(graph, selected, actions, rule="coverage", remaining=1.0):
    if rule in ("coverage", "heuristic"):
        return min(actions, key=lambda action: graph.heuristic_key(selected, action))
    if rule == "degree":
        return min(actions, key=lambda action: (sum(graph.log_degree[i] for i in action.added),
                   -sum(graph.log_degree[i] for i in action.removed), graph.priority_key(action)))
    if rule == "random":
        return min(actions, key=graph.priority_key)
    if rule in ("rule", "canonical"):
        return actions[0]
    raise ValueError(f"Unknown selection rule: {rule}")


def repair(graph, selected, *, rounds=4, cap=48, anchor=None, deadline=inf, extend_first=True):
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
        actions = improving_actions(graph, result.work, cap=cap, anchor=anchor, deadline=deadline)
        result.candidate_counts.append(len(actions))
        if perf_counter() >= deadline:
            result.deadline_hit = True
            return result
        if not actions:
            result.exhausted = True
            return result
        candidate = graph.apply(result.work, actions[0])
        if perf_counter() >= deadline:
            result.deadline_hit = True
            return result
        result.work = candidate
        result.exchanges += 1
    result.capped = True
    return result


def rollout(graph, selected, action, *, horizon=2, repair_rounds=4,
            improving_cap=48, candidate_cap=32, deadline=inf):
    """The same finite macro is used for labels and deployment; history is local."""
    if horizon not in (1, 2):
        raise ValueError("horizon must be 1 or 2")
    result = RolloutResult(selected, selected, 0)
    attempted = []
    if perf_counter() >= deadline:
        result.deadline_hit = True
        return result
    work = graph.apply_move(selected, action)
    if perf_counter() >= deadline:
        result.deadline_hit = True
        return result
    attempted.append(action)
    anchor = action.added[0] if len(action.removed) == 2 else None
    for stage in range(horizon):
        if stage:
            strict = improving_actions(graph, work, cap=1, deadline=deadline)
            if perf_counter() >= deadline:
                result.deadline_hit = True
                break
            if not strict:
                pool = candidate_actions(graph, work, cap=candidate_cap, tabu=(action,), deadline=deadline)
                if not pool.completed or perf_counter() >= deadline:
                    result.deadline_hit = True
                    break
                if not pool.actions:
                    break
                second = choose_action(graph, work, pool.actions)
                candidate = graph.apply_move(work, second)
                if perf_counter() >= deadline:
                    result.deadline_hit = True
                    break
                work = candidate
                attempted.append(second)
        segment = repair(graph, work, rounds=repair_rounds, cap=improving_cap,
                         anchor=anchor if stage == 0 else None, deadline=deadline)
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


def action_values(graph, selected, actions, *, return_rollouts=False, **kwargs):
    results = [rollout(graph, selected, action, **kwargs) for action in actions]
    values = [result.gain for result in results]
    return (values, results) if return_rollouts else values


def platform_search(graph, selected, *, choose="coverage", outer_rounds=8, initial_rounds=3,
                    horizon=2, candidate_cap=32, improving_cap=48, repair_rounds=4,
                    anchored=False, history=(), deadline=inf, on_state=None):
    """Initial v2 rule23, then at most outer_rounds rule or neutral macro rounds.

    A scorer has signature choose(graph, selected, actions, remaining_fraction).
    on_state(graph, selected, pool, history, outer_index, remaining_fraction)
    observes decision states before selection without changing search state.
    """
    limited = isfinite(deadline)
    work_deadline = deadline - 0.002 if limited else deadline
    initial_start = perf_counter() if limited else 0.0
    initial = repair(graph, selected, rounds=initial_rounds, cap=improving_cap,
                     deadline=work_deadline, extend_first=False) if initial_rounds else RepairResult(selected)
    initial_seconds = perf_counter() - initial_start if limited else 0.0
    round_seconds, score_units = [], []
    macro_seconds = [max(0.002, initial_seconds * horizon)]
    result = SearchResult(initial.work, initial.work, tuple(history)[-2:], selected.bit_count(),
                          initial.work.bit_count(), exchanges=initial.exchanges,
                          deadline_hit=initial.deadline_hit)
    for step in range(outer_rounds):
        if result.deadline_hit or perf_counter() >= work_deadline:
            result.deadline_hit = True
            break
        round_start = perf_counter() if limited else 0.0
        strict = improving_actions(graph, result.work, cap=1, deadline=work_deadline)
        if perf_counter() >= work_deadline:
            result.deadline_hit = True
            break
        before = result.work
        remaining = (outer_rounds - step) / outer_rounds
        if strict:
            segment = repair(graph, before, rounds=repair_rounds, cap=improving_cap, deadline=work_deadline)
            result.work = segment.work
            result.exchanges += segment.exchanges
            result.deadline_hit = segment.deadline_hit
            row = dict(kind="repair", exchanges=segment.exchanges, capped=segment.capped)
        else:
            if limited:
                predicted_round = 1.25 * max(round_seconds[-2:]) if round_seconds else 0.003
                if perf_counter() + predicted_round >= work_deadline:
                    result.deadline_hit = True
                    break
            pool = candidate_actions(graph, before, cap=candidate_cap, anchored=anchored,
                                     tabu=result.history, deadline=work_deadline)
            if not pool.completed or perf_counter() >= work_deadline:
                result.deadline_hit = True
                break
            if on_state is not None:
                on_state(graph, before, pool, result.history, step, remaining)
            if not pool.actions:
                break
            if limited:
                count = len(pool.actions)
                if callable(choose) and count > 1:
                    unit = max(score_units[-2:]) if score_units else 0.0006 * graph.num_nodes / 250
                    predicted_score = unit * count
                else:
                    predicted_score = 0.0005 if callable(choose) else 0.0001
                predicted_macro = max(macro_seconds[-2:])
                if perf_counter() + 1.25 * (predicted_score + predicted_macro) >= work_deadline:
                    result.deadline_hit = True
                    break
                score_start = perf_counter()
            action = (choose(graph, before, pool.actions, remaining) if callable(choose)
                      else choose_action(graph, before, pool.actions, choose))
            if limited:
                actual_score = perf_counter() - score_start
                if callable(choose) and count > 1:
                    score_units.append(actual_score / count)
                if perf_counter() + 1.25 * predicted_macro >= work_deadline:
                    result.deadline_hit = True
                    break
                macro_start = perf_counter()
            macro = rollout(graph, before, action, horizon=horizon, repair_rounds=repair_rounds,
                            improving_cap=improving_cap, candidate_cap=candidate_cap, deadline=work_deadline)
            result.work = macro.work
            result.history = (result.history + macro.neutral_actions)[-2:]
            result.exchanges += macro.exchanges
            result.neutral_decisions += len(macro.attempted_actions)
            result.deadline_hit = macro.deadline_hit
            row = dict(kind="neutral", candidate_count=len(pool.actions), total_count=pool.total_count,
                       exchanges=macro.exchanges, neutral_steps=len(macro.attempted_actions),
                       returned_to_start=macro.returned_to_start, rolled_back=macro.rolled_back,
                       ended_below_start=macro.ended_below_start, gain=macro.gain, capped=macro.capped)
            if limited:
                actual_macro = perf_counter() - macro_start
                actual_round = perf_counter() - round_start
                macro_seconds.append(actual_macro)
                round_seconds.append(actual_round)
                row.update(round_time_s=actual_round, score_time_s=actual_score,
                           predicted_score_time_s=predicted_score, rollout_time_s=actual_macro)
            if macro.best.bit_count() > result.best.bit_count():
                result.best = macro.best
        if result.work.bit_count() > result.best.bit_count():
            result.best = result.work
        result.outer_steps += 1
        row.update(outer=step, before_size=before.bit_count(), after_size=result.work.bit_count())
        result.trace.append(row)
    return result
