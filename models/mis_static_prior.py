"""Two-action calibration and graph-centered static MIS bias."""

import torch
from torch import nn

from models import edge2adj
from v7_structure import V7Topology


STATIC_MODES = ("static_constants", "static_centered")
STATIC_FEATURES = ("log_degree", "neighbor_log_degree_mean", "neighbor_log_degree_std", "density", "size")


def cheap_features(energy_adj, ptr):
    degree = energy_adj.crow_indices().diff().float()
    log_degree = degree.log1p()
    # Energy adjacency carries half-weight edges. Both moments share one SpMM.
    moments = (energy_adj.float() @ torch.stack((log_degree, log_degree.square()), dim=1))
    moments = moments * (2 / degree.clamp_min(1)[:, None])
    mean, second = moments.unbind(dim=1)
    std = (second - mean.square()).clamp_min(0).sqrt()
    dense, sizes = [], []
    for start, stop in zip(ptr, ptr[1:]):
        n = stop - start
        dense.append((degree[start:stop].sum() / max(1, n * (n-1))).expand(n))
        sizes.append(torch.full_like(degree[start:stop], n / 300))
    return torch.stack((log_degree, mean, std, torch.cat(dense), torch.cat(sizes)), dim=1)


class StaticPrior(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.constants = nn.Parameter(torch.zeros(2))
        self.centered = mode == "static_centered"
        if self.centered:
            self.alpha = nn.Parameter(torch.zeros(()))
            self.context = nn.Sequential(nn.Linear(5, 16), nn.SiLU(), nn.Linear(16, 2))
            self.register_buffer("feature_mean", torch.zeros(5))
            self.register_buffer("feature_std", torch.ones(5))

    def topology(self, graph, energy_adj):
        ptr = getattr(graph, "graph_ptr", (0, graph.num_nodes))
        features = None
        if self.centered:
            raw = cheap_features(energy_adj, ptr)
            features = ((raw - self.feature_mean) / self.feature_std).clamp(-5, 5)
        return V7Topology(edge2adj(graph.edge_index, graph.num_nodes), features, ptr, (), 1)

    def action_bias(self, topology, _limit):
        constants = 0.25 * self.constants.tanh()[None, :]
        if not self.centered or topology.features.size(0) == 0:
            return constants
        h = self.context(topology.features).tanh()
        centered = torch.cat([h[a:b] - h[a:b].mean(dim=0) for a, b in zip(topology.ptr, topology.ptr[1:]) if b > a])
        return constants + 0.125 * self.alpha.tanh() * centered
