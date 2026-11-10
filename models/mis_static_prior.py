"""Two-action calibration and graph-centered static MIS bias."""

import json
import torch
from torch import nn

from models import edge2adj
from v7_structure import FEATURES, V7Topology, v7_topology


STATIC_MODES = ("static_constants", "static_centered", "static_random", "static_oracle")
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


def graph_center(values, ptr, *, normalize=False):
    parts = []
    for a, b in zip(ptr, ptr[1:]):
        if b > a:
            part = values[a:b] - values[a:b].mean(0)
            if normalize:
                peak = part.abs().amax()
                part = part / torch.where(peak > 0, peak, torch.ones_like(peak))
            parts.append(part)
    return torch.cat(parts)


def bias_stats(bias):
    values = bias.float()
    means = values.mean(0)
    return dict(bias_rms=float(values.square().mean().sqrt()),
                node_bias_rms=float((values - means).square().mean().sqrt()),
                add_mean=float(means[0]), drop_mean=float(means[1]))


def random_bias(n, seed, rms, device):
    direction = torch.randn((n, 2), generator=torch.Generator().manual_seed(seed)).to(device)
    direction = direction - direction.mean(0)
    return direction * (rms / direction.square().mean().sqrt().clamp_min(1e-12))


def oracle_bias(n, selected, amplitude, device):
    add = torch.full((n,), -amplitude, device=device)
    add[selected] = amplitude
    return torch.stack((add, -add), dim=1)


class StaticPrior(nn.Module):
    def __init__(self, mode, *, constant_limit=.25, node_limit=.125, feature_dim=5,
                 bias_seed=1017, oracle_path=None, encoder=None, order_seed=1017, views=1, peak_normalize=False):
        super().__init__()
        self.mode = mode
        if min(constant_limit, node_limit) < 0:
            raise ValueError("Static limits must be nonnegative")
        self.constant_limit, self.node_limit = constant_limit, node_limit
        self.bias_seed = bias_seed
        self.order_seed, self.views = order_seed, views
        self.peak_normalize = peak_normalize
        self.feature_schema = STATIC_FEATURES if feature_dim == 5 else FEATURES
        self.feature_dim = feature_dim
        self.feature_drop = None
        self.offline_features = False
        self.encoder = encoder
        self.constants = nn.Parameter(torch.zeros(2))
        self.centered = mode == "static_centered"
        self.oracle = None
        if mode == "static_oracle":
            if oracle_path is None:
                raise ValueError("static_oracle requires --static_oracle_path")
            self.oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
        if self.centered:
            self.alpha = nn.Parameter(torch.zeros(()))
            if encoder is None:
                self.context = nn.Sequential(nn.Linear(feature_dim, 16), nn.SiLU(), nn.Linear(16, 2))
            self.register_buffer("feature_mean", torch.zeros(feature_dim))
            self.register_buffer("feature_std", torch.ones(feature_dim))

    def normalize(self, raw):
        values = ((raw - self.feature_mean) / self.feature_std).clamp(-5, 5)
        if self.feature_drop is not None:
            values[:, self.feature_drop] = 0
        return values

    def topology(self, graph, energy_adj, *, timings=None, graph_id=None):
        ptr = getattr(graph, "graph_ptr", (0, graph.num_nodes))
        features = None
        if self.centered:
            if self.feature_dim == 8 or self.encoder is not None:
                return v7_topology(graph, self.normalize, self.order_seed,
                                   order=self.encoder.order if self.encoder is not None else None,
                                   views=self.views, timings=timings, sparse=True,
                                   offline=self.training or self.offline_features)
            features = self.normalize(cheap_features(energy_adj, ptr))
        elif self.mode == "static_random":
            features = random_bias(graph.num_nodes, self.bias_seed + int(graph.idx), 1, graph.edge_index.device)
        elif self.mode == "static_oracle":
            selected = self.oracle[graph_id if graph_id in self.oracle else str(graph.idx)]
            features = oracle_bias(graph.num_nodes, selected, 1, graph.edge_index.device)
        return V7Topology(edge2adj(graph.edge_index, graph.num_nodes), features, ptr, (), 1)

    def direction(self, topology, *, normalize=False):
        h = self.context(topology.features).float().tanh() if self.encoder is None else self.encoder.precompute(topology)
        return graph_center(h, topology.ptr, normalize=normalize)

    def action_bias(self, topology, limits=None):
        constant_limit, node_limit = (self.constant_limit, self.node_limit) if limits is None else limits
        constants = constant_limit * self.constants.tanh()[None, :]
        if self.mode in ("static_random", "static_oracle"):
            return constants + node_limit * topology.features
        if not self.centered or topology.features.size(0) == 0:
            return constants
        return constants + node_limit * self.alpha.tanh() * self.direction(topology, normalize=self.peak_normalize)
