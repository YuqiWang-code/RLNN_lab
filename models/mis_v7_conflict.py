"""Conflict-certified release regions for the MIS plateau search (TCCR, plan B).

For a legal independent set ``S`` and an unselected node ``v`` the blocker set is
``B(v) = N(v) & S``.  Choosing a set ``U`` of selected nodes to release gives the
release set and the local reoptimisation window::

    R(U) = {v not in S : B(v) <= U}
    W(U) = U | R(U)

``R(U)`` touches nothing outside ``U`` among the selected nodes, so the merged
set ``(S \\ U) | T`` is legal for every legal ``T`` inside ``G[W(U)]``.  Solving
the local problem exactly gives ``|T| >= |U|``, hence the reoptimisation never
shrinks the incumbent.  The interesting question is how often ``alpha(G[W(U)])``
exceeds ``|U|``, i.e. whether a joint ``k -> k+1`` action exists that no
single-flip or 1-to-2 platform action can reach.

The exact local solve is a maximum independent set on a small induced subgraph,
implemented as a maximum clique on the complement with Tomita-style colouring
bounds.  Everything is plain Python over integer bitsets.
"""
from dataclasses import dataclass


def _bits(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def blocker_map(selected, neighbours, num_nodes):
    """``v -> B(v)`` for every unselected node, ``B(v) = N(v) & S``."""
    blockers = {}
    for v in range(num_nodes):
        if selected >> v & 1:
            continue
        blockers[v] = neighbours[v] & selected
    return blockers


def release_set(blockers, u_mask):
    """Nodes ``v`` with ``B(v) <= U``; nodes with ``B(v) = 0`` are excluded.

    ``B(v) = 0`` means ``v`` could be added without releasing anything, so the
    incumbent was not single-flip maximal.  Such nodes are reported separately by
    the caller instead of being counted as a release.
    """
    release = 0
    free = 0
    for v, blockers_v in blockers.items():
        if not blockers_v:
            free |= 1 << v
        elif not blockers_v & ~u_mask:
            release |= 1 << v
    return release, free


def regions(selected, neighbours, num_nodes, size):
    """Yield ``(U, W, R)`` for every ``size``-subset of ``selected``.

    ``R`` is the release set and ``W = U | R`` the local window.  Regions with an
    empty release set cannot improve and are skipped.
    """
    blockers = blocker_map(selected, neighbours, num_nodes)
    released = []
    for v, blockers_v in blockers.items():
        if blockers_v:
            released.append((v, blockers_v))
    nodes = list(_bits(selected))
    if size > len(nodes):
        return
    import itertools

    for combination in itertools.combinations(nodes, size):
        u_mask = 0
        for v in combination:
            u_mask |= 1 << v
        release = 0
        for v, blockers_v in released:
            if not blockers_v & ~u_mask:
                release |= 1 << v
        if release.bit_count() < 2:
            # One released node can always be left alone or exchanged for one of
            # U, so alpha <= |U|; two is the smallest window that can grow.
            continue
        yield u_mask, u_mask | release, release


def greedy_independent_set(mask, neighbours):
    """Maximal independent set by lowest current degree; a valid lower bound."""
    chosen = 0
    remaining = mask
    while remaining:
        best = -1
        best_degree = None
        for v in _bits(remaining):
            degree = (neighbours[v] & remaining).bit_count()
            if best_degree is None or degree < best_degree:
                best = v
                best_degree = degree
        chosen |= 1 << best
        remaining &= ~(neighbours[best] | (1 << best))
    return chosen


def clique_cover_bound(mask, complement):
    """Upper bound on a clique of ``complement`` by greedy colouring.

    Each colour class is independent in ``complement``, hence a clique in the
    original graph, so the colour count bounds every clique in ``complement``.
    """
    remaining = mask
    colours = 0
    while remaining:
        colours += 1
        available = remaining
        while available:
            v = (available & -available).bit_length() - 1
            available &= ~complement[v] & ~(1 << v)
            remaining &= ~(1 << v)
    return colours


def bounds(mask, neighbours):
    """``(LB, UB)`` for ``alpha(G[mask])``; both are cheap and valid."""
    if not mask:
        return 0, 0
    return greedy_independent_set(mask, neighbours).bit_count(), clique_cover_bound(
        mask, _complement(mask, neighbours))


def _complement(mask, neighbours):
    complement = [0] * len(neighbours)
    for v in _bits(mask):
        complement[v] = mask & ~neighbours[v] & ~(1 << v)
    return complement


def maximum_independent_set(mask, neighbours, cap=60):
    """Exact ``alpha(G[mask])`` and one maximiser.

    Returns ``(size, set_mask, exact)``.  Subgraphs larger than ``cap`` nodes fall
    back to the greedy construction so that a pathological region cannot stall
    the census; ``exact`` is ``False`` in that case.
    """
    if not mask:
        return 0, 0, True
    if mask.bit_count() > cap:
        return greedy_independent_set(mask, neighbours).bit_count(), 0, False
    complement = _complement(mask, neighbours)
    best = [0, 0]
    path = []

    def expand(candidates):
        order = []
        colours = []
        remaining = candidates
        colour = 0
        while remaining:
            colour += 1
            available = remaining
            while available:
                v = (available & -available).bit_length() - 1
                order.append(v)
                colours.append(colour)
                available &= ~complement[v] & ~(1 << v)
                remaining &= ~(1 << v)
        for index in range(len(order) - 1, -1, -1):
            if len(path) + colours[index] <= best[0]:
                return
            v = order[index]
            path.append(v)
            expand(candidates & complement[v])
            path.pop()
            candidates &= ~(1 << v)
            if len(path) + 1 > best[0]:
                best[0] = len(path) + 1
                best[1] = 0
                for u in path:
                    best[1] |= 1 << u
                best[1] |= 1 << v

    expand(mask)
    return best[0], best[1], True


@dataclass(frozen=True)
class Region:
    """One released subset, its window and the exact local optimum."""

    u_mask: int
    window: int
    release: int
    alpha: int
    gain: int
    exact: bool
    lower: int
    upper: int

    @property
    def removed(self):
        return self.u_mask.bit_count()


def solve_region(u_mask, release, neighbours):
    """Exact local solve of ``G[U | R]`` and the resulting gain."""
    window = u_mask | release
    alpha, _, exact = maximum_independent_set(window, neighbours)
    removed = u_mask.bit_count()
    lower, upper = (removed, removed) if not release else bounds(window, neighbours)
    return Region(u_mask=u_mask, window=window, release=release, alpha=alpha,
                  gain=alpha - removed, exact=exact, lower=lower, upper=upper)


def census(selected, neighbours, num_nodes, sizes=(2, 3, 4), cap=60):
    """Enumerate regions by size and keep the best gain per size.

    Returns ``{size: {"regions": n, "improving": n, "best": Region | None,
    "best_gain": int, "exact": bool}}``.
    """
    result = {}
    for size in sizes:
        count = 0
        improving = 0
        best = None
        exact = True
        for u_mask, window, release in regions(selected, neighbours, num_nodes, size):
            count += 1
            region = solve_region(u_mask, release, neighbours)
            exact = exact and region.exact
            if region.gain > 0:
                improving += 1
            if best is None or region.gain > best.gain:
                best = region
        result[size] = dict(regions=count, improving=improving, best=best,
                            best_gain=best.gain if best else 0, exact=exact)
    return result
