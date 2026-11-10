"""Check legal plateau actions, bounded repair, anchors and shared rollout labels."""
from itertools import combinations
from pathlib import Path
import random
import sys
from time import perf_counter
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))
from mis_v3_search import (STATS_DIM, SearchGraph, action_values, candidate_actions,
                           improving_actions, node_mask, nodes, platform_search,
                           repair, rollout)
import mis_v2_exchange
import mis_v3_search


def graph_from(size, pairs, seed=17):
    directed = [(u, v) for a, b in pairs for u, v in ((a, b), (b, a))]
    edges = torch.tensor(directed, dtype=torch.long).reshape(-1, 2).T
    return SearchGraph(size, edges, seed=seed)


def enumerate_check():
    rng = random.Random(31)
    moves = strict_moves = matched = 0
    for _ in range(18):
        graph = graph_from(8, [edge for edge in combinations(range(8), 2) if rng.random() < 0.35])
        for selected in range(1 << 8):
            if not graph.feasible(selected) or graph.extend(selected) != selected:
                continue
            outside = list(nodes(graph.all_nodes ^ selected))
            pool = candidate_actions(graph, selected, anchored=True, cap=None)
            expected = {(tuple(nodes(graph.neighbors[v] & selected)), (v,)) for v in outside
                        if (graph.neighbors[v] & selected).bit_count() in (1, 2)}
            assert {(a.removed, a.added) for a in pool.actions} == expected
            assert pool.completed and pool.total_count == len(expected)
            for action in pool.actions:
                successor = graph.apply_move(selected, action)
                assert graph.feasible(successor)
                assert successor.bit_count() == selected.bit_count() - len(action.removed) + 1
                moves += 1
            expected_strict = set()
            for size in (1, 2):
                for removed in combinations(nodes(selected), size):
                    retained = selected & ~node_mask(removed)
                    for added in combinations(outside, size + 1):
                        if graph.feasible(retained | node_mask(added)):
                            expected_strict.add((removed, added))
            actual = improving_actions(graph, selected, cap=None, union=True)
            assert {(a.removed, a.added) for a in actual} == expected_strict
            strict_moves += len(actual)
            if not actual and pool.actions:
                neutral = [a for a in pool.actions if len(a.removed) == 1]
                if not neutral:
                    continue
                values, labels = action_values(graph, selected, neutral, return_rollouts=True)
                index = max(range(len(values)), key=values.__getitem__)
                selected_action = neutral[index]
                deployed = platform_search(graph, selected, initial_rounds=0, outer_rounds=1,
                                           choose=lambda g, s, actions, remaining: selected_action)
                assert deployed.work == labels[index].work
                assert deployed.best == labels[index].best
                assert deployed.best.bit_count() - selected.bit_count() == values[index]
                assert graph.feasible(deployed.work) and graph.feasible(deployed.best)
                matched += 1
    assert moves and strict_moves and matched
    return moves, strict_moves, matched


def anchor_and_history_check():
    graph = graph_from(3, [(0, 2), (1, 2)])
    selected = node_mask((0, 1))
    action = candidate_actions(graph, selected, anchored=True).actions[0]
    successor = graph.apply_move(selected, action)
    assert graph.apply(successor, improving_actions(graph, successor)[0]) == selected
    protected = repair(graph, successor, anchor=2)
    assert protected.work == successor and protected.exhausted
    limited = rollout(graph, selected, action, horizon=1)
    assert limited.ended_below_start and limited.rolled_back
    assert limited.work == selected and limited.gain == 0 and not limited.neutral_actions
    extended = rollout(graph, selected, action, horizon=2)
    assert extended.work == selected and extended.gain == 0

    graph = graph_from(2, [(0, 1)])
    selected = node_mask((0,))
    action = candidate_actions(graph, selected).actions[0]
    label = rollout(graph, selected, action, horizon=2)
    assert label.work == node_mask((1,)) and len(label.attempted_actions) == 1
    assert not candidate_actions(graph, label.work, tabu=(action,)).actions
    assert len(candidate_actions(graph, label.work).actions) == 1


def budget_and_features_check():
    pairs = [(i, 6 + 2 * i + j) for i in range(6) for j in range(2)]
    graph = graph_from(18, pairs)
    selected = node_mask(range(6))
    limited = repair(graph, selected, rounds=4)
    assert limited.exchanges == 4 and limited.capped and not limited.exhausted
    deployed = platform_search(graph, selected, initial_rounds=0, outer_rounds=2, repair_rounds=2)
    assert deployed.exchanges == 4 and deployed.outer_steps == 2
    assert all(row["kind"] == "repair" for row in deployed.trace)
    assert deployed.best.bit_count() == selected.bit_count() + 4
    expired = platform_search(graph, selected, deadline=perf_counter() - 1)
    assert expired.best == selected and expired.deadline_hit and expired.outer_steps == 0
    assert not candidate_actions(graph, selected, deadline=perf_counter() - 1).completed

    complete = candidate_actions(graph, selected, cap=None)
    capped = candidate_actions(graph, selected, cap=6)
    assert capped.total_count == 12 and len(capped.actions) == 6
    assert capped.actions == candidate_actions(graph, selected, cap=6).actions
    expected_first = sorted(complete.actions, key=lambda a: graph.heuristic_key(selected, a))[:3]
    assert capped.actions[:3] == expected_first
    stats = torch.tensor(graph.stats_features(selected, capped.actions))
    node_features = torch.tensor(graph.node_features(selected))
    assert stats.shape == (6, STATS_DIM) and STATS_DIM == 24
    assert node_features.shape == (18, 8)
    assert torch.isfinite(stats).all() and torch.isfinite(node_features).all()
    assert torch.all(node_features[:6, 0] == 1)
    assert torch.allclose(node_features[6:, 1], torch.full((12,), 1 / 3))
    assert torch.all(node_features[6:, 3] == 1)
    assert torch.allclose(node_features[:6, 7], torch.full((6,), torch.log(torch.tensor(2.)).item()))


def deadline_prediction_check():
    graph = graph_from(250, [edge for start in range(0, 250, 10)
                            for edge in combinations(range(start, start + 10), 2)])
    selected = node_mask(range(0, 250, 10))

    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    scores = []

    def candidates(*args, **kwargs):
        clock.now += 0.001
        return candidate_actions(*args, **kwargs)

    def repairing(*args, **kwargs):
        clock.now += 0.001
        return repair(*args, **kwargs)

    def continuing(*args, **kwargs):
        clock.now += 0.002
        return rollout(*args, **kwargs)

    def score(graph, selected, actions, remaining):
        scores.append(clock.now)
        clock.now += 0.030
        return actions[0]

    with patch.object(mis_v3_search, "perf_counter", clock), \
            patch.object(mis_v2_exchange, "perf_counter", clock), \
            patch.object(mis_v3_search, "candidate_actions", candidates), \
            patch.object(mis_v3_search, "repair", repairing), \
            patch.object(mis_v3_search, "rollout", continuing):
        too_short = platform_search(graph, selected, choose=score, horizon=1, deadline=0.020)
        assert not scores and too_short.best == selected and too_short.deadline_hit
        clock.now = 0.0
        timed = platform_search(graph, selected, choose=score, horizon=1, deadline=0.130)
        assert len(scores) == 3 and timed.outer_steps == 3 and timed.deadline_hit
        assert clock.now < 0.128 and graph.feasible(timed.best)
        assert all(row["score_time_s"] >= 0.029 for row in timed.trace)
        clock.now = 0.0
        scores.clear()
        unlimited = platform_search(graph, selected, choose=score, horizon=1)
        assert len(scores) == 8 and unlimited.outer_steps == 8 and not unlimited.deadline_hit


if __name__ == "__main__":
    torch.set_num_threads(1)
    counts = enumerate_check()
    anchor_and_history_check()
    budget_and_features_check()
    deadline_prediction_check()
    print(f"v3 checks passed: {counts[0]} plateau actions, {counts[1]} improving actions, "
          f"{counts[2]} paired label/deployment states; anchors, history, caps, features and deadline prediction")
