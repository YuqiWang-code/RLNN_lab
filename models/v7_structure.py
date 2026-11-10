"""Static MIS graph features and independent component orders."""

from functools import lru_cache
from time import perf_counter
from typing import NamedTuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components, dijkstra
import torch

from models import edge2adj


FEATURES = ("log_degree", "neighbor_log_degree_mean", "neighbor_log_degree_std",
            "core", "hub_distance", "two_hop_fraction", "density", "size")


class V7Topology(NamedTuple):
    normalized: torch.Tensor
    features: torch.Tensor | None
    ptr: tuple
    orders: tuple
    views: int
    cached_action_bias: torch.Tensor | None = None


def core_numbers(adj, degree):
    # Batagelj–Zaversnik peeling, with one constant-time bucket move per edge.
    core = degree.tolist()
    vertices = sorted(range(len(core)), key=core.__getitem__)
    position = [0] * len(core)
    boundary = [0] * (max(core, default=0) + 1)
    for i, node in enumerate(vertices):
        position[node] = i
        boundary[core[node]] += 1
    offset = 0
    for k, count in enumerate(boundary):
        boundary[k], offset = offset, offset + count
    ptr, neighbors = adj.indptr.tolist(), adj.indices.tolist()
    for node in vertices:
        for other in neighbors[ptr[node]:ptr[node + 1]]:
            k = core[other]
            if k > core[node]:
                first = boundary[k]
                j = position[other]
                peer = vertices[first]
                vertices[j], vertices[first] = peer, other
                position[peer], position[other] = j, first
                boundary[k] += 1
                core[other] -= 1
    return np.asarray(core)


def peel_small_graph(matrix, degree):
    remaining = np.ones(len(degree), dtype=bool)
    current = degree.copy()
    core = np.empty_like(degree)
    shell = 0
    while remaining.any():
        shell = max(shell, current[remaining].min())
        peel = remaining & (current <= shell)
        core[peel] = shell
        remaining[peel] = False
        current -= matrix[peel].sum(axis=0, dtype=current.dtype)
    return core


def small_graph_features(adj, degree):
    matrix = adj.toarray()
    core = peel_small_graph(matrix, degree)
    dense = matrix.astype(np.float32)
    reach = (dense @ dense > 0) | matrix
    return core, reach.sum(axis=1) - reach.diagonal()


def structure(edges, n, order_seed, *, order=None, views=1, tie_keys=None, timings=None, sparse=False):
    tick = perf_counter()
    if n == 0:
        return np.empty((0, len(FEATURES)), dtype=np.float32), ()
    adj = csr_matrix((np.ones(edges.shape[1], dtype=bool), edges), shape=(n, n))
    degree = np.diff(adj.indptr)
    log = np.log1p(degree)
    divisor = np.maximum(degree, 1)
    mean = (adj @ log) / divisor
    std = np.sqrt(np.maximum((adj @ log**2) / divisor - mean**2, 0))
    if n <= 512 and not sparse:
        core, reach_count = small_graph_features(adj, degree)
    else:
        core = peel_small_graph(adj.toarray(), degree) if n <= 512 else core_numbers(adj, degree)
        reach = (adj @ adj).maximum(adj)
        reach_count = np.diff(reach.indptr) - reach.diagonal()
    two_hop = reach_count / n
    count, component = connected_components(adj, directed=False)
    random = np.random.default_rng(order_seed)
    keys = random.random((views + 1, n)) if tie_keys is None else tie_keys
    parts = (np.arange(n),) if count == 1 else tuple(np.flatnonzero(component == c) for c in range(count))
    hubs = [nodes[np.lexsort((keys[0, nodes], -core[nodes], -degree[nodes]))[0]] for nodes in parts]
    if sparse:
        distance = dijkstra(adj, indices=hubs, directed=True, unweighted=True, min_only=True)
    else:
        distance = [0] * n
        for hub in hubs:
            queue, predecessor = breadth_first_order(adj, hub, directed=True)
            previous = predecessor.tolist()
            for node in queue.tolist()[1:]:
                distance[node] = distance[previous[node]] + 1
        distance = np.asarray(distance)
    features = np.column_stack((log, mean, std, core, distance, two_hop,
                                np.full(n, degree.sum() / max(1, n * (n - 1))),
                                np.full(n, n / 300))).astype(np.float32)
    if timings is not None:
        timings["static_features_s"] += perf_counter() - tick
        tick = perf_counter()
    orders = {}
    if order is not None:
        for nodes in parts:
            sequences = []
            for jitter in keys[1:]:
                if order == "shuffle":
                    ranking = np.argsort(jitter[nodes])
                elif order == "degree":
                    ranking = np.lexsort((jitter[nodes], -degree[nodes]))
                else:
                    ranking = np.lexsort((jitter[nodes], -degree[nodes], -core[nodes], distance[nodes]))
                sequences.append(nodes[ranking])
            orders.setdefault(len(nodes), []).append(np.stack(sequences))
    if timings is not None:
        timings["ordering_s"] += perf_counter() - tick
    return features, tuple(np.concatenate(group) for group in orders.values())


@lru_cache(maxsize=1024)
def offline_structure(n, edges, order_seed, order=None, views=1):
    """Cache raw v2 features and orders independently of checkpoint normalization."""
    return structure(np.frombuffer(edges, dtype=np.int64).reshape(2, -1), n, order_seed,
                     order=order, views=views, sparse=True)


def v7_topology(graph, normalizer, order_seed, *, order=None, views=1, timings=None, sparse=False, offline=False):
    device = graph.edge_index.device
    n = graph.num_nodes
    ptr = getattr(graph, "graph_ptr", (0, n))
    ids = getattr(graph, "graph_ids", (int(getattr(graph, "idx", 0)),))
    edges = graph.edge_index.cpu().numpy()
    features, orders = [], []
    for index, (start, end) in enumerate(zip(ptr, ptr[1:])):
        local = edges if len(ptr) == 2 else edges[:, (edges[0] >= start) & (edges[0] < end)] - start
        if offline:
            tick = perf_counter() if timings is not None else None
            values, sequences = offline_structure(end - start, local.astype(np.int64, copy=False).tobytes(),
                                                  order_seed + ids[index], order, views)
            if tick is not None:
                timings["cached_structure_s"] = timings.get("cached_structure_s", 0.) + perf_counter() - tick
        else:
            values, sequences = structure(local, end - start, order_seed + ids[index], order=order,
                                         views=views, timings=timings, sparse=sparse)
        features.append(values)
        orders.extend(torch.as_tensor(seq + start, device=device) for seq in sequences)
    values = torch.as_tensor(np.concatenate(features), device=device)
    normalized = edge2adj(graph.edge_index, n)
    return V7Topology(normalized, normalizer(values), ptr, tuple(orders), views)
