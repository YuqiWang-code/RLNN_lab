"""Static MIS graph features and independent component orders."""

from time import perf_counter
from typing import NamedTuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
import torch

from models import edge2adj


FEATURES = ("log_degree", "neighbor_log_degree_mean", "neighbor_log_degree_std",
            "core", "hub_distance", "two_hop_fraction", "density", "size")


class V7Topology(NamedTuple):
    normalized: torch.Tensor
    features: torch.Tensor
    ptr: tuple
    orders: tuple
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


def structure(edges, n, order_seed, *, order=None, views=1, tie_keys=None, timings=None):
    tick = perf_counter()
    if n == 0:
        return np.empty((0, len(FEATURES)), dtype=np.float32), ()
    adj = csr_matrix((np.ones(edges.shape[1], dtype=bool), edges), shape=(n, n))
    degree = np.diff(adj.indptr)
    log = np.log1p(degree)
    divisor = np.maximum(degree, 1)
    mean = (adj @ log) / divisor
    std = np.sqrt(np.maximum((adj @ log**2) / divisor - mean**2, 0))
    core = core_numbers(adj, degree)
    reach = (adj @ adj).maximum(adj)
    two_hop = (np.diff(reach.indptr) - reach.diagonal()) / n
    count, component = connected_components(adj, directed=False)
    random = np.random.default_rng(order_seed)
    keys = random.random((views + 1, n)) if tie_keys is None else tie_keys
    parts = (np.arange(n),) if count == 1 else tuple(np.flatnonzero(component == c) for c in range(count))
    hubs = [nodes[np.lexsort((keys[0, nodes], -core[nodes], -degree[nodes]))[0]] for nodes in parts]
    if count == 1:
        distance = shortest_path(adj, directed=False, unweighted=True, indices=hubs[0])
    else:
        distance = np.empty(n)
        for nodes, hub in zip(parts, hubs):
            distance[nodes] = shortest_path(adj, directed=False, unweighted=True, indices=hub)[nodes]
    features = np.column_stack((log, mean, std, core, distance, two_hop,
                                np.full(n, degree.sum() / max(1, n * (n - 1))),
                                np.full(n, n / 300))).astype(np.float32)
    if timings is not None:
        timings["static_features_s"] += perf_counter() - tick
        tick = perf_counter()
    orders = []
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
            orders.append(np.stack(sequences))
    if timings is not None:
        timings["ordering_s"] += perf_counter() - tick
    return features, tuple(orders)


def v7_topology(graph, normalizer, order_seed, *, order=None, views=1, timings=None):
    device = graph.edge_index.device
    n = graph.num_nodes
    ptr = getattr(graph, "graph_ptr", (0, n))
    ids = getattr(graph, "graph_ids", (int(getattr(graph, "idx", 0)),))
    edges = graph.edge_index.cpu().numpy()
    features, orders = [], []
    for index, (start, end) in enumerate(zip(ptr, ptr[1:])):
        local = edges if len(ptr) == 2 else edges[:, (edges[0] >= start) & (edges[0] < end)] - start
        values, sequences = structure(local, end - start, order_seed + ids[index], order=order,
                                     views=views, timings=timings)
        features.append(values)
        orders.extend(torch.as_tensor(seq + start, device=device) for seq in sequences)
    values = torch.as_tensor(np.concatenate(features), device=device)
    normalized = edge2adj(graph.edge_index, n)
    return V7Topology(normalized, normalizer(values), ptr, tuple(orders))
