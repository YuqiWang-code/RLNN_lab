"""Bounded improving exchanges on a legal MIS solution."""
from itertools import combinations, islice
from math import fsum, inf, log1p
from time import perf_counter
from typing import NamedTuple


def nodes(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def node_mask(indices):
    return sum(1 << node for node in indices)


class Exchange(NamedTuple):
    removed: tuple
    added: tuple
    release_count: int


class ExchangeGraph:
    def __init__(self, num_nodes, edge_index):
        self.neighbors = [0] * num_nodes
        for source, target in zip(*edge_index.tolist()):
            self.neighbors[source] |= 1 << target
        self.all_nodes = (1 << num_nodes) - 1
        self.log_degree = [log1p(row.bit_count()) for row in self.neighbors]
        self.greedy_order = sorted(range(num_nodes), key=self.log_degree.__getitem__)

    def feasible(self, selected):
        return all(not self.neighbors[node] & selected for node in nodes(selected))

    def extend(self, selected):
        available = self.all_nodes ^ selected
        for node in nodes(selected):
            available &= ~self.neighbors[node]
        for node in self.greedy_order:
            bit = 1 << node
            if available & bit:
                selected |= bit
                available &= ~(bit | self.neighbors[node])
        return selected

    def _exchanges(self, selected, size, deadline):
        singles = {}
        doubles = {}
        for node in nodes(self.all_nodes ^ selected):
            blockers = self.neighbors[node] & selected
            count = blockers.bit_count()
            if count == 1:
                singles[blockers] = singles.get(blockers, 0) | (1 << node)
            elif size == 2 and count == 2:
                doubles[blockers] = doubles.get(blockers, 0) | (1 << node)

        for removed in combinations(nodes(selected), size):
            if perf_counter() >= deadline:
                return
            if size == 1:
                released = singles.get(1 << removed[0], 0)
            else:
                first, second = (1 << node for node in removed)
                released = singles.get(first, 0) | singles.get(second, 0) | doubles.get(first | second, 0)
            release_count = released.bit_count()
            while released:
                if perf_counter() >= deadline:
                    return
                first = released & -released
                released ^= first
                u = first.bit_length() - 1
                compatible = released & ~self.neighbors[u]
                if size == 1:
                    for v in nodes(compatible):
                        yield Exchange(removed, (u, v), release_count)
                else:
                    while compatible:
                        second = compatible & -compatible
                        compatible ^= second
                        v = second.bit_length() - 1
                        for w in nodes(compatible & ~self.neighbors[v]):
                            yield Exchange(removed, (u, v, w), release_count)

    def actions(self, selected, *, cap=None, larger=False, deadline=inf):
        actions = list(islice(self._exchanges(selected, 1, deadline), cap))
        if not actions and larger:
            actions = list(islice(self._exchanges(selected, 2, deadline), cap))
        return actions

    def apply(self, selected, action):
        candidate = (selected & ~node_mask(action.removed)) | node_mask(action.added)
        candidate = self.extend(candidate)
        if candidate.bit_count() <= selected.bit_count() or not self.feasible(candidate):
            raise ValueError("Exchange must produce a larger independent set")
        return candidate

    def coverage(self, selected, action):
        available = self.all_nodes & ~(selected | node_mask(action.added))
        neighborhoods = [self.neighbors[node] & available for node in action.added]
        union, shared = 0, 0
        for row in neighborhoods:
            shared |= union & row
            union |= row
        return union.bit_count(), shared.bit_count(), [row.bit_count() for row in neighborhoods]

    def heuristic(self, selected, action):
        covered, _, _ = self.coverage(selected, action)
        return covered, -action.release_count

    def features(self, selected, action):
        added_degree = [self.log_degree[node] for node in action.added]
        covered, shared, remaining = self.coverage(selected, action)
        remaining = [log1p(count) for count in remaining]
        return (fsum(self.log_degree[node] for node in action.removed), fsum(added_degree),
                max(added_degree) - min(added_degree), log1p(action.release_count),
                fsum(remaining), max(remaining) - min(remaining), log1p(covered),
                log1p(shared), len(action.removed))


def improve(graph, selected, *, cap, rounds, larger=False, deadline=inf, choose=None, repeat_choice=False):
    """Choose the first action, then use the shared degree/coverage rule."""
    changes = 0
    candidate_count = 0
    for step in range(rounds):
        if perf_counter() >= deadline:
            break
        actions = graph.actions(selected, cap=cap, larger=larger, deadline=deadline)
        if step == 0:
            candidate_count = len(actions)
        if not actions or perf_counter() >= deadline:
            break
        if choose is not None and (step == 0 or repeat_choice):
            action = choose(graph, selected, actions)
        else:
            action = min(actions, key=lambda a: graph.heuristic(selected, a))
        candidate = graph.apply(selected, action)
        if perf_counter() >= deadline:
            break
        selected = candidate
        changes += 1
    return selected, changes, candidate_count


def action_values(graph, selected, actions, *, cap, rounds, larger=False):
    values = []
    for action in actions:
        successor = graph.apply(selected, action)
        final, _, _ = improve(graph, successor, cap=cap, rounds=rounds - 1, larger=larger)
        values.append(final.bit_count())
    return values
