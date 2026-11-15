"""Bitset large-neighbourhood moves: exact repair of a ball around a solution.

The deployed loop only ever moves one node at a time, and its archive keeps the
best state, so it can never cross a barrier.  The exchange probe showed that
every move up to four points is already closed.  This module builds the next
rung: take a *region* (a BFS ball), drop the solution's points inside it, and
solve the induced subproblem **exactly** -- the replacement may add more points
than it removes, which is a move of arbitrary size.

Everything is Python integers used as bitsets; a ball of radius two on an RB
graph has a few dozen nodes, where branch and bound is microseconds.
"""


def neighbourhood_masks(edges, n):
    """Adjacency as a list of integer bitsets (undirected)."""
    neighbours = [0] * n
    for u, v in zip(edges[0].tolist(), edges[1].tolist()):
        neighbours[u] |= 1 << v
        neighbours[v] |= 1 << u
    return neighbours


def ball(neighbours, centre, radius):
    """Bitset of nodes within ``radius`` hops of ``centre`` (inclusive)."""
    seen = 1 << centre
    frontier = seen
    for _ in range(radius):
        grown = 0
        rest = frontier
        while rest:
            bit = rest & -rest
            rest ^= bit
            grown |= neighbours[bit.bit_length() - 1]
        frontier = grown & ~seen
        if not frontier:
            break
        seen |= frontier
    return seen


def exact_mis(mask, neighbours, memo=None):
    """Maximum independent set of the subgraph induced by ``mask``.

    Returns ``(size, solution_mask)``.  Branches on the vertex of highest degree
    inside the remaining mask, which keeps the tree small on sparse balls.
    """
    if memo is None:
        memo = {}
    if mask == 0:
        return 0, 0
    hit = memo.get(mask)
    if hit is not None:
        return hit
    best_vertex, best_degree = -1, -1
    rest = mask
    while rest:
        bit = rest & -rest
        rest ^= bit
        vertex = bit.bit_length() - 1
        degree = bin(neighbours[vertex] & mask).count("1")
        if degree > best_degree:
            best_degree, best_vertex = degree, vertex
    vertex = best_vertex
    size_out, set_out = exact_mis(mask & ~(1 << vertex), neighbours, memo)
    size_in, set_in = exact_mis(mask & ~neighbours[vertex] & ~(1 << vertex), neighbours, memo)
    size_in += 1
    set_in |= 1 << vertex
    result = (size_in, set_in) if size_in >= size_out else (size_out, set_out)
    memo[mask] = result
    return result


def repair(selected, region, neighbours):
    """Best replacement of ``selected & region`` inside ``region``.

    Nodes that touch a selected point outside the region are excluded, so the
    repaired set stays independent against the untouched part of the solution.
    Returns ``(gain, replacement_mask)``.
    """
    outside = selected & ~region
    blocked = 0
    rest = outside
    while rest:
        bit = rest & -rest
        rest ^= bit
        blocked |= neighbours[bit.bit_length() - 1]
    allowed = region & ~blocked
    size, solution = exact_mis(allowed, neighbours)
    removed = bin(selected & region).count("1")
    return size - removed, solution


def apply_repair(selected, region, replacement):
    """Swap the region's points for the repaired set."""
    return (selected & ~region) | replacement
