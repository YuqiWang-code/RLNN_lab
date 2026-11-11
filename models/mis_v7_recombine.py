"""Exact recombination of two independent sets through a bipartite difference graph.

For two legal independent sets ``a`` and ``b`` of the same graph write::

    I = a & b          common nodes        (independent on its own)
    L = a \\ b          nodes only in a
    R = b \\ a          nodes only in b

``a`` and ``b`` are independent, so ``G[L | R]`` has no edge inside ``L`` and no
edge inside ``R``: it is bipartite.  No node of ``I`` touches ``L | R`` either.
Ko nig's theorem then gives the exact optimum of the difference region::

    alpha(G[L | R]) = |L| + |R| - nu(G[L | R])

with ``nu`` the maximum matching size.  A maximum independent set of the
difference region is the complement of a minimum vertex cover, which is built
from the matching by alternating reachability.  The recombined set is::

    S_new = I | T*,   |S_new| = |I| + |L| + |R| - nu >= max(|a|, |b|)

The guarantee is local: it is exact inside ``a | b`` only, it says nothing about
the rest of the graph, and it does not force a strict improvement.
Everything here is plain Python over integer bitsets.  The operator therefore
carries no torch dependency and can be unit tested, run inside the CUDA census,
and dropped into a deployment path unchanged.
"""
from dataclasses import dataclass


def bits(mask):
    """Yield the node indices of a bitset from low to high."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def adjacency_masks(num_nodes, edges):
    """Per-node neighbour bitsets from an ``edge_index`` style iterable."""
    neighbours = [0] * num_nodes
    for u, v in edges:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        neighbours[u] |= 1 << v
        neighbours[v] |= 1 << u
    return neighbours


def is_independent(mask, neighbours):
    """True when no two nodes of ``mask`` share an edge."""
    rest = mask
    while rest:
        node = rest & -rest
        index = node.bit_length() - 1
        if neighbours[index] & rest & ~node:
            return False
        rest ^= node
    return True


def is_maximal(mask, neighbours, num_nodes):
    """True when no single addition keeps the set independent."""
    for v in range(num_nodes):
        if not mask >> v & 1 and not neighbours[v] & mask:
            return False
    return True


def _maximum_matching(left, right, neighbours):
    """Maximum matching between two disjoint node bitsets (Kuhn's algorithm).

    The difference regions here have at most a few dozen nodes, so the simple
    augmenting-path formulation is used instead of a batched one: it always
    terminates, and it is fast enough that a phase-based Hopcroft-Karp would only
    add failure modes.  Returns ``(match_left, match_right)`` dictionaries.
    """
    adjacency = {u: neighbours[u] & right for u in bits(left)}
    match_right = {}

    def augment(u, seen):
        for v in bits(adjacency[u]):
            if v in seen:
                continue
            seen.add(v)
            mate = match_right.get(v)
            if mate is None or augment(mate, seen):
                match_right[v] = u
                return True
        return False

    for u in adjacency:
        augment(u, set())
    match_left = {u: v for v, u in match_right.items()}
    return match_left, match_right


def matching_size(left, right, neighbours):
    """Maximum matching size ``nu`` of the bipartite difference graph."""
    match_left, _ = _maximum_matching(left, right, neighbours)
    return len(match_left)


def minimum_vertex_cover(left, right, neighbours):
    """Koenig construction: minimum vertex cover of the difference graph.

    Alternating reachability starts from the unmatched ``L`` vertices and walks
    non-matching edges ``L -> R`` and matching edges ``R -> L``.  The cover is
    ``(L \\ Z_L) | (R & Z_R)``.
    """
    match_left, match_right = _maximum_matching(left, right, neighbours)
    reach_left = 0
    reach_right = 0
    stack = [u for u in bits(left) if u not in match_left]
    for u in stack:
        reach_left |= 1 << u
    while stack:
        u = stack.pop()
        for v in bits(neighbours[u] & right):
            if match_left.get(u) == v:
                continue
            if reach_right >> v & 1:
                continue
            reach_right |= 1 << v
            mate = match_right.get(v)
            if mate is not None and not reach_left >> mate & 1:
                reach_left |= 1 << mate
                stack.append(mate)
    return (left & ~reach_left) | (right & reach_right), len(match_left)


@dataclass(frozen=True)
class Recombination:
    """Result of recombining two independent sets."""

    candidate: int
    common: int
    left: int
    right: int
    matching: int
    size: int
    best_parent: int
    gain: int          # candidate size - best parent size, always >= 0

    @property
    def union_loss(self):
        """Size lost relative to ``a | b``; equals the matching size ``nu``."""
        return self.matching

    @property
    def left_size(self):
        return self.left.bit_count()

    @property
    def right_size(self):
        return self.right.bit_count()


def recombine(a, b, neighbours):
    """Exact recombination of ``a`` and ``b`` inside ``a | b``.

    Returns a :class:`Recombination`.  ``candidate`` is guaranteed to be an
    independent set whenever ``a`` and ``b`` are, and it is never smaller than
    the larger parent.
    """
    common = a & b
    left = a & ~b
    right = b & ~a
    cover, matching = minimum_vertex_cover(left, right, neighbours)
    difference_best = (left | right) & ~cover
    candidate = common | difference_best
    size = candidate.bit_count()
    best_parent = max(a.bit_count(), b.bit_count())
    expected = common.bit_count() + left.bit_count() + right.bit_count() - matching
    if size != expected:  # pragma: no cover - guards the theorem itself
        raise AssertionError(f"recombination size {size} != {expected}")
    if not is_independent(candidate, neighbours):  # pragma: no cover
        raise AssertionError("recombination returned a conflicting set")
    return Recombination(candidate=candidate, common=common, left=left, right=right,
                         matching=matching, size=size, best_parent=best_parent,
                         gain=size - best_parent)


def polish(mask, neighbours, num_nodes=None):
    """Greedy flip fixed point of the A.3 rule, restricted to feasible states.

    ``decode_gradient_flip`` never decreases the energy, and on a legal set the
    only admissible flip is ``0 -> 1`` into a node without selected neighbours.
    Repeating that flip until none is available is therefore a monotone polish
    of a legal set; the result is maximal in the single-flip neighbourhood.
    """
    if num_nodes is None:
        num_nodes = len(neighbours)
    while True:
        for v in range(num_nodes):
            if mask >> v & 1 or neighbours[v] & mask:
                continue
            mask |= 1 << v
            break
        else:
            return mask


def pair_gain(a, b, neighbours, use_polish=False):
    """Gain of the recombined pair over its better parent."""
    result = recombine(a, b, neighbours)
    if use_polish:
        polished = polish(result.candidate, neighbours)
        return polished.bit_count() - result.best_parent
    return result.gain


def best_recombination(solutions, neighbours, use_polish=False):
    """Oracle over all unordered pairs of a solution list.

    ``best_gain`` is measured against the best *single* solution of the list,
    because that is the frozen baseline the operator has to beat (the A.3 argmax).
    ``pair_gains`` keeps each pair's own gain relative to its better parent, which
    is larger whenever a good mix comes from two mediocre parents; the count of
    pairs that beat the global baseline is therefore reported separately by the
    caller.

    Returns ``(best_candidate, best_gain, best_pair, pair_gains)``.
    """
    gains = []
    best_gain = 0
    best_candidate = max(solutions, key=int.bit_count)
    baseline = best_candidate.bit_count()
    best_pair = None
    for (i, j) in pairs(len(solutions)):
        result = recombine(solutions[i], solutions[j], neighbours)
        if use_polish:
            candidate = polish(result.candidate, neighbours)
            gains.append(candidate.bit_count() - result.best_parent)
        else:
            candidate = result.candidate
            gains.append(result.gain)
        gain = candidate.bit_count() - baseline
        if gain > best_gain:
            best_gain = gain
            best_candidate = candidate
            best_pair = (i, j)
    return best_candidate, best_gain, best_pair, gains


def pairs(count):
    """All unordered index pairs of ``range(count)``."""
    return [(i, j) for i in range(count) for j in range(i + 1, count)]


def select_pairs(solutions, strategy, budget=None):
    """Cheap pair orderings for the non-oracle arms of the census.

    ``strategy`` is one of ``random``, ``hamming``, ``top_size`` and
    ``matching_cost``.  ``budget`` caps the number of returned pairs.
    """
    import random

    count = len(solutions)
    ordered = pairs(count)
    if strategy == "random":
        rng = random.Random(0)
        rng.shuffle(ordered)
    elif strategy == "hamming":
        ordered.sort(key=lambda pair: (solutions[pair[0]] ^ solutions[pair[1]]).bit_count(),
                     reverse=True)
    elif strategy == "top_size":
        order = sorted(range(count), key=lambda index: solutions[index].bit_count(), reverse=True)
        ordered = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]
    elif strategy == "matching_cost":
        # cheapest first: small difference region first, as a proxy for operator cost
        ordered.sort(key=lambda pair: (solutions[pair[0]] ^ solutions[pair[1]]).bit_count())
    else:
        raise ValueError(f"unknown pair strategy: {strategy}")
    return ordered if budget is None else ordered[:budget]
