"""Fast screening of the recombination oracle for deadline-limited use.

The exact operator in :mod:`mis_v7_recombine` costs about 300 us per pair in pure
Python: fine for an offline census (58 ms for all 190 pairs), too slow for the
20 ms slice a 0.5 s deadline can spare.  Almost all pairs have nothing to gain,
so the expensive part can be screened away:

1. the graph adjacency is packed into one dense boolean matrix (single vectorised
   ``unpackbits``);
2. every pair's bipartite difference block is a small fancy-index slice of it;
3. all blocks are stacked into one block-diagonal sparse matrix and handed to
   ``scipy.sparse.csgraph.maximum_bipartite_matching`` in a single call, giving
   every pair's exact matching size ``nu``;
4. the exact operator then runs only on the pairs that can strictly improve
   (``max(|L|, |R|) - nu > 0``), usually none or one.

Because ``nu`` is exact and the final candidate comes from the same code, the
screened oracle returns exactly what the exhaustive oracle would, in a few
milliseconds instead of tens.
"""
import numpy as np

import mis_v7_recombine as recombine

try:  # scipy is present in the project environment; keep a correct fallback anyway
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
except ImportError:  # pragma: no cover - exercised only on minimal installs
    csr_matrix = None
    maximum_bipartite_matching = None


def adjacency_matrix(neighbours):
    """Dense boolean adjacency from the integer bitsets used everywhere else."""
    count = len(neighbours)
    words = (count + 63) // 64
    packed = np.zeros((count, words), dtype=np.uint64)
    for node, mask in enumerate(neighbours):
        if not mask:
            continue
        encoded = int(mask).to_bytes(words * 8, "little")
        packed[node] = np.frombuffer(encoded, dtype="<u8")
    bits = np.unpackbits(packed.view(np.uint8), axis=1, bitorder="little")
    return bits[:, :count].astype(bool)


def solution_index(solutions, count):
    """Boolean presence matrix and per-solution node index arrays."""
    present = np.zeros((len(solutions), count), dtype=bool)
    index = []
    for row, mask in enumerate(solutions):
        nodes = np.fromiter(recombine.bits(mask), dtype=np.int64)
        index.append(nodes)
        present[row, nodes] = True
    return present, index


def difference_blocks(solutions, matrix, present=None, index=None):
    """``(left, right, block)`` for every unordered pair, in :func:`pairs` order.

    ``left``/``right`` come from boolean masks on the per-solution index arrays,
    which keeps the per-pair Python work to a handful of numpy calls.
    """
    if present is None or index is None:
        present, index = solution_index(solutions, matrix.shape[0])
    blocks = []
    for i, j in recombine.pairs(len(solutions)):
        left = index[i][~present[j][index[i]]]
        right = index[j][~present[i][index[j]]]
        if left.size and right.size:
            block = matrix[np.ix_(left, right)]
        else:
            block = np.zeros((left.size, right.size), dtype=bool)
        blocks.append((left, right, block))
    return blocks


def matching_sizes(blocks):
    """Exact maximum matching size per block, in one batched call."""
    if maximum_bipartite_matching is None:  # pragma: no cover
        return [_kuhn(block) for _, _, block in blocks]
    rows, cols = [], []
    row_offsets = []
    row_total = col_total = 0
    for left, right, block in blocks:
        row_offsets.append(row_total)
        if block.any():
            ii, jj = np.nonzero(block)
            rows.append(ii + row_total)
            cols.append(jj + col_total)
        row_total += block.shape[0]
        col_total += block.shape[1]
    if not rows:
        return [0] * len(blocks)
    data = np.ones(sum(chunk.size for chunk in rows), dtype=np.int8)
    matrix = csr_matrix((data, (np.concatenate(rows), np.concatenate(cols))),
                        shape=(max(row_total, 1), max(col_total, 1)))
    matched = maximum_bipartite_matching(matrix, perm_type="column")
    sizes = []
    for index, (left, right, block) in enumerate(blocks):
        start = row_offsets[index]
        sizes.append(int((matched[start:start + block.shape[0]] >= 0).sum()))
    return sizes


def _kuhn(block):  # pragma: no cover - only without scipy
    """Pure-Python fallback used when scipy is unavailable."""
    if not block.size:
        return 0
    match_right = {}
    adjacency = [list(np.nonzero(row)[0]) for row in block]

    def augment(u, seen):
        for v in adjacency[u]:
            if v in seen:
                continue
            seen.add(v)
            mate = match_right.get(v)
            if mate is None or augment(mate, seen):
                match_right[v] = u
                return True
        return False

    for u in range(len(adjacency)):
        augment(u, set())
    return len(match_right)


def pair_gains(solutions, blocks, sizes, orders=None):
    """Exact gain of every pair over the global best single solution.

    ``common + |L| + |R| - nu`` is the recombined size, and ``nu`` is exact, so
    this is not a bound: it is the gain itself.  Only the winning pair needs the
    exact operator to materialise its node set.
    """
    if orders is None:
        orders = recombine.pairs(len(solutions))
    baseline = max(mask.bit_count() for mask in solutions)
    gains = []
    for index, (left, right, _) in enumerate(blocks):
        i, j = orders[index]
        common = (solutions[i] & solutions[j]).bit_count()
        gains.append(common + int(left.size) + int(right.size) - sizes[index] - baseline)
    return gains


def oracle(solutions, neighbours, matrix=None):
    """Screened exact oracle: same answer as ``recombine.best_recombination``.

    Returns ``(candidate, gain, best_pair, exact_calls)``.  ``sizes`` is exact, so
    exactly one exact operator call is needed to build the winning candidate.
    """
    if matrix is None:
        matrix = adjacency_matrix(neighbours)
    present, index = solution_index(solutions, matrix.shape[0])
    blocks = difference_blocks(solutions, matrix, present, index)
    sizes = matching_sizes(blocks)
    orders = recombine.pairs(len(solutions))
    gains = pair_gains(solutions, blocks, sizes, orders)
    if not gains or max(gains) <= 0:
        return max(solutions, key=int.bit_count), 0, None, 0
    index = max(range(len(gains)), key=lambda k: gains[k])
    i, j = orders[index]
    result = recombine.recombine(solutions[i], solutions[j], neighbours)
    baseline = max(mask.bit_count() for mask in solutions)
    return result.candidate, result.size - baseline, (i, j), 1


def oracle_with_deadline(solutions, neighbours, matrix=None, deadline=None):
    """Same as :func:`oracle` but abandons the exact call if the deadline passes.

    The screening already ranks the pairs, so a truncated run keeps the best legal
    parent rather than returning something infeasible.
    """
    import time

    if matrix is None:
        matrix = adjacency_matrix(neighbours)
    present, index = solution_index(solutions, matrix.shape[0])
    blocks = difference_blocks(solutions, matrix, present, index)
    sizes = matching_sizes(blocks)
    orders = recombine.pairs(len(solutions))
    gains = pair_gains(solutions, blocks, sizes, orders)
    best = max(solutions, key=int.bit_count)
    if not gains or max(gains) <= 0:
        return best, 0, None, 0
    index = max(range(len(gains)), key=lambda k: gains[k])
    i, j = orders[index]
    if deadline is not None and time.perf_counter() >= deadline:
        return best, 0, None, 0
    result = recombine.recombine(solutions[i], solutions[j], neighbours)
    baseline = best.bit_count()
    return result.candidate, result.size - baseline, (i, j), 1
