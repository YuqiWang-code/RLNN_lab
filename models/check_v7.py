"""Correctness checks for the TCCR operators (recombination and release regions).

Everything here runs on CPU without checkpoints or data caches, so it can be
executed on any machine before a GPU census:

    python models/check_v7.py

Checks, in order:

1. recombination against brute force on random small graphs;
2. recombination is feasible, never smaller than its better parent, and its size
   equals the Koenig formula;
3. ``polish`` keeps a legal set legal, maximal, and never shrinks it;
4. the exact region solver matches brute force on random induced subgraphs;
5. the hand-built 3-to-4 example from the plan is detected as one improving
   4-window while no single-flip action exists.
"""
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mis_v7_conflict as conflict
import mis_v7_recombine as recombine


def random_graph(rng, num_nodes, probability):
    edges = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)
             if rng.random() < probability]
    return recombine.adjacency_masks(num_nodes, edges)


def random_legal_set(rng, num_nodes, neighbours):
    order = list(range(num_nodes))
    rng.shuffle(order)
    mask = 0
    for v in order:
        if not neighbours[v] & mask:
            mask |= 1 << v
    return mask


def brute_force_alpha(mask, neighbours):
    nodes = list(recombine.bits(mask))
    if not nodes:
        return 0
    if len(nodes) > 20:
        raise ValueError("brute force is only used for tiny subgraphs")
    best = 0
    for size in range(len(nodes), 0, -1):
        for combination in itertools.combinations(nodes, size):
            candidate = 0
            for v in combination:
                candidate |= 1 << v
            if recombine.is_independent(candidate, neighbours):
                return size
    return best


def check_recombination(trials=200):
    rng = random.Random(20261111)
    checked = 0
    for _ in range(trials):
        num_nodes = rng.randint(6, 12)
        neighbours = random_graph(rng, num_nodes, rng.uniform(0.1, 0.45))
        a = random_legal_set(rng, num_nodes, neighbours)
        b = random_legal_set(rng, num_nodes, neighbours)
        result = recombine.recombine(a, b, neighbours)
        assert recombine.is_independent(result.candidate, neighbours)
        assert result.size >= max(a.bit_count(), b.bit_count())
        union = a | b
        common = a & b
        difference = union & ~common
        expected = common.bit_count() + brute_force_alpha(difference, neighbours)
        assert result.size == expected, (result.size, expected)
        assert result.union_loss == result.matching
        assert result.size == union.bit_count() - result.matching
        checked += 1
    # 200 random pairs of legal sets, all consistent with brute force
    return checked


def check_oracle_accounting(trials=60):
    """best_recombination must report the gain over the global best solution."""
    rng = random.Random(23)
    for _ in range(trials):
        num_nodes = rng.randint(8, 18)
        neighbours = random_graph(rng, num_nodes, rng.uniform(0.05, 0.25))
        solutions = []
        for _ in range(20):
            order = list(range(num_nodes))
            rng.shuffle(order)
            mask = 0
            for v in order:
                if not neighbours[v] & mask:
                    mask |= 1 << v
            solutions.append(recombine.polish(mask, neighbours))
        candidate, gain, pair, pair_gains = recombine.best_recombination(solutions, neighbours)
        base = max(mask.bit_count() for mask in solutions)
        assert candidate.bit_count() - base == gain, (candidate.bit_count(), base, gain)
        assert gain <= max(pair_gains or [0])
        assert recombine.is_independent(candidate, neighbours)
    return trials


def check_polish(trials=100):
    rng = random.Random(7)
    for _ in range(trials):
        num_nodes = rng.randint(8, 16)
        neighbours = random_graph(rng, num_nodes, rng.uniform(0.1, 0.4))
        legal = random_legal_set(rng, num_nodes, neighbours)
        polished = recombine.polish(legal, neighbours)
        assert recombine.is_independent(polished, neighbours)
        assert polished.bit_count() >= legal.bit_count()
        assert recombine.is_maximal(polished, neighbours, num_nodes)
    return trials


def check_regions(trials=120):
    rng = random.Random(11)
    for _ in range(trials):
        num_nodes = rng.randint(8, 14)
        neighbours = random_graph(rng, num_nodes, rng.uniform(0.1, 0.35))
        selected = recombine.polish(random_legal_set(rng, num_nodes, neighbours), neighbours)
        for size in (2, 3):
            for u_mask, window, release in conflict.regions(selected, neighbours, num_nodes, size):
                alpha, _, exact = conflict.maximum_independent_set(window, neighbours)
                assert exact
                assert window == u_mask | release
                assert not release & selected
                # the window may freely be re-optimised: nothing outside it is touched
                assert not (release & selected)
                assert alpha == brute_force_alpha(window, neighbours), (alpha, window.bit_count())
                # releasing keeps the complement of U free of conflicts with R
                outside = selected & ~u_mask
                for v in recombine.bits(release):
                    assert not neighbours[v] & outside
    return trials


def check_three_to_four():
    """The plan's example: replacing three tight blockers by four joint releases."""
    # nodes 0,1,2 are the current set; 3,4,5,6 are each blocked by all three
    num_nodes = 7
    edges = [(u, v) for u in (0, 1, 2) for v in (3, 4, 5, 6)]
    neighbours = recombine.adjacency_masks(num_nodes, edges)
    selected = 0b111
    assert recombine.is_independent(selected, neighbours)
    assert recombine.is_maximal(selected, neighbours, num_nodes)
    result = conflict.census(selected, neighbours, num_nodes, sizes=(3,))
    assert result[3]["regions"] == 1, result[3]
    assert result[3]["improving"] == 1, result[3]
    best = result[3]["best"]
    assert best.removed == 3 and best.alpha == 4 and best.gain == 1
    merged = (selected & ~best.u_mask) | conflict.maximum_independent_set(
        best.window, neighbours)[1]
    assert merged.bit_count() == 4
    assert recombine.is_independent(merged, neighbours)
    return True


def main():
    results = {}
    results["recombination_pairs"] = check_recombination()
    results["oracle_cases"] = check_oracle_accounting()
    results["polish_cases"] = check_polish()
    results["region_cases"] = check_regions()
    results["three_to_four"] = check_three_to_four()
    for name, value in results.items():
        print(f"{name}: {value}")
    print("all v7 operator checks passed")


if __name__ == "__main__":
    main()
