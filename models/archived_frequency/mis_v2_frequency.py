"""MIS exchange-conditioned graph filters and residual GCN adapters."""
from typing import NamedTuple

import torch
from torch import nn

from mis_v1 import state_features
from models import GCN, edge2adj


SEGF_MODES = ("segf_mlp", "segf_stats", "segf_static1", "segf_static2",
              "segf_exchange", "segf_high", "segf_fixed", "segf")


class FrequencyTopology(NamedTuple):
    normalized: torch.Tensor
    degree: torch.Tensor
    log_degree: torch.Tensor


def frequency_topology(edge_index, adj, nodes):
    degree = adj.crow_indices().diff().to(adj.dtype).view(-1, 1)
    return FrequencyTopology(edge2adj(edge_index, nodes), degree, degree.log1p())


def neighbor_sum(adj, signals):
    return 2 * (adj @ signals.flatten(1)).reshape_as(signals)


def exchange_state(x, adj, topology, *, exchange=True):
    selected = neighbor_sum(adj, x)
    release = ((x == 0) & (selected == 1)).to(x.dtype)
    released = x * neighbor_sum(adj, release) if exchange else torch.zeros_like(x)
    features = torch.stack((x, selected.clamp(max=2), topology.log_degree.expand_as(x),
                            released.log1p()), dim=-1)
    return features, release, released


def lazy_walk(features, adj, topology):
    degree = topology.degree.unsqueeze(-1)
    neighbors = neighbor_sum(adj, features) / degree.clamp_min(1)
    return (features + torch.where(degree > 0, neighbors, features)) * 0.5


def exchange_band(features, x, release, released, adj):
    # Factor the two endpoint masks around the sparse edge reduction.
    selected = x.unsqueeze(-1)
    freeable = release.unsqueeze(-1)
    source = torch.cat((features * freeable, features * selected), dim=-1)
    to_selected, to_freeable = neighbor_sum(adj, source).chunk(2, dim=-1)
    degree = (released + release).unsqueeze(-1)
    incoming = selected * to_selected + freeable * to_freeable
    average = incoming / degree.clamp_min(1)
    return torch.where(degree > 0, features - average, 0)


def encoder(channels, width, hidden):
    return nn.Sequential(nn.Linear(channels, width), nn.SiLU(),
                         nn.Linear(width, hidden), nn.LayerNorm(hidden))


class FrequencyAdapter(nn.Module):
    def __init__(self, hidden, beta, mode):
        super().__init__()
        self.beta = beta
        self.mode = mode
        self.low = self.high = None
        if mode in ("segf_mlp", "segf_stats", "segf_exchange"):
            channels = {"segf_mlp": 1, "segf_stats": 4, "segf_exchange": 5}[mode]
            # Match the full two-branch adapter with a raw-state MLP.
            width = round((67 * hidden + 715) / (hidden + 2)) if mode == "segf_mlp" else 32
            self.low = encoder(channels, width, hidden)
        elif mode == "segf_high":
            self.high = encoder(12, 32, hidden)
        else:
            self.low = encoder(4 if mode == "segf_static1" else 8, 32, hidden)
            self.high = encoder({"segf_static1": 4, "segf_static2": 8}.get(mode, 12), 32, hidden)
        if self.low is not None:
            self.alpha_low = nn.Parameter(torch.tensor(0.01))
        if self.high is not None:
            self.alpha_high = nn.Parameter(torch.tensor(0.01))
        self.gates = nn.Linear(4, 2) if mode == "segf" else None

    def branches(self, x, adj, topology):
        if self.mode == "segf_mlp":
            return self.alpha_low * self.low(x.unsqueeze(-1)), None, None
        if self.mode == "segf_stats":
            features, _ = state_features(x, adj, self.beta)
            return self.alpha_low * self.low(features), None, None

        use_exchange = self.mode in ("segf_exchange", "segf_high", "segf_fixed", "segf")
        features, release, released = exchange_state(x, adj, topology, exchange=use_exchange)
        if self.mode == "segf_exchange":
            siblings = release * neighbor_sum(adj, released)
            inputs = torch.cat((features, siblings.log1p().unsqueeze(-1)), dim=-1)
            return self.alpha_low * self.low(inputs), None, None

        first = lazy_walk(features, adj, topology)
        low_input = [first]
        high_input = [features - first]
        if self.mode != "segf_static1":
            second = lazy_walk(first, adj, topology)
            low_input.append(second)
            high_input.append(first - second)
        if use_exchange:
            high_input.append(exchange_band(features, x, release, released, adj))
        gates = self.gates(features).sigmoid() if self.gates is not None else None
        weight_low, weight_high = gates.unbind(-1) if gates is not None else (None, None)
        low = self.alpha_low * self.low(torch.cat(low_input, -1)) if self.low is not None else None
        high = self.alpha_high * self.high(torch.cat(high_input, -1))
        if gates is not None:
            low = low * weight_low.unsqueeze(-1)
            high = high * weight_high.unsqueeze(-1)
        elif low is not None:
            low, high = 0.5 * low, 0.5 * high
        return low, high, gates


class SEGF(GCN):
    def __init__(self, hidden, layers, beta, mode):
        super().__init__(1, hidden, layers, 1)
        self.adapter = FrequencyAdapter(hidden, beta, mode)

    def forward(self, x, edge_index, adj, topology=None):
        if topology is None:
            topology = frequency_topology(edge_index, adj, x.size(0))
        hidden = self.initial_project(x.unsqueeze(-1))
        low, high, _ = self.adapter.branches(x, adj, topology)
        if low is not None:
            hidden = hidden + low
        if high is not None:
            hidden = hidden + high
        return self.project_hidden(hidden, topology.normalized).squeeze(-1)
