"""Degree-conditioned structural compensation for the RLNN flip policy.

The baseline policy is ``GCN(1, 128, 5, 1)``: its only input channel is the binary
state, so degree, neighbourhood occupancy and the combinatorial gain of a flip all
have to be reconstructed through mean aggregation. High-degree nodes are exactly
where that reconstruction is weakest and where MIS cares most.

Modes (all zero-initialised, so step 0 equals B bit for bit):

``degree``      per-layer additive ``W_l phi(deg)`` in the residual stream (5 extra adds per step).
``gate``        bounded degree-conditioned scaling of the aggregated message.
``gain``        explicit combinatorial-gain residual on the flip logits, tanh-gated.
``all``         the three above together.
``degree_out``  one rank-one ``phi(deg) @ w`` term on the readout: a single broadcast add per step.
``all_out``     ``degree_out`` plus a gain residual with a learnable positive scale.
"""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from models import GCN, edge2adj


STRUCTURE_MODES = ("none", "degree", "gate", "gain", "all", "degree_out", "all_out")
DEGREE_FREQUENCIES = (1., 2., 3.)
GAIN_FEATURES = 6
READOUT_MODES = ("degree_out", "all_out")
DEGREE_FEATURE_DIM = 2 + 2 * len(DEGREE_FREQUENCIES)


class StructureTopology(NamedTuple):
    normalized: torch.Tensor
    degree: torch.Tensor
    degree_feature: torch.Tensor
    degree_centered: torch.Tensor
    ptr: tuple
    cached_bias: tuple | None = None
    cached_scale: tuple | None = None
    cached_readout: torch.Tensor | None = None


def degree_features(degree, ptr):
    """Fourier encoding of the graph-normalised log degree plus its centred value."""
    segments = [(a, b) for a, b in zip(ptr, ptr[1:]) if b > a]
    sizes = torch.as_tensor([max(1, b - a) for a, b in segments], dtype=degree.dtype, device=degree.device)
    scale = torch.repeat_interleave(sizes.log1p().clamp_min(1e-6),
                                   torch.as_tensor([b - a for a, b in segments], device=degree.device))
    unit = degree.log1p() / scale
    parts = [unit, unit.square()]
    for frequency in DEGREE_FREQUENCIES:
        angle = torch.pi * frequency * unit
        parts.extend((angle.cos(), angle.sin()))
    return torch.stack(parts, dim=1), torch.cat([unit[a:b] - unit[a:b].mean() for a, b in segments])


def structure_topology(edge_index, normalized, num_nodes, ptr=None):
    """Static per-node structural quantities, computed once per solve."""
    ptr = tuple(ptr) if ptr is not None else (0, num_nodes)
    degree = torch.bincount(edge_index[0], minlength=num_nodes).to(torch.float32)
    feature, centered = degree_features(degree, ptr)
    return StructureTopology(normalized, degree, feature, centered, ptr)


def zero_linear(layer):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class StructureCompensation(nn.Module):
    """The compensation terms, split out so ablations only flip a string."""

    def __init__(self, hidden, layers, mode, beta=1.02):
        super().__init__()
        if mode not in STRUCTURE_MODES or mode == "none":
            raise ValueError(f"unknown structure mode {mode}")
        self.mode = mode
        self.beta = beta
        self.use_degree = mode in ("degree", "all")
        self.use_gate = mode in ("gate", "all")
        self.use_gain = mode in ("gain", "all")
        self.use_readout = mode in READOUT_MODES
        self.use_scaled_gain = mode == "all_out"
        if self.use_degree:
            self.degree_proj = nn.ModuleList([zero_linear(nn.Linear(DEGREE_FEATURE_DIM, hidden))
                                              for _ in range(layers)])
        if self.use_gate:
            self.gate = nn.Parameter(torch.zeros(layers))
        if self.use_gain:
            self.gain_gate = nn.Parameter(torch.zeros(()))
            self.gain_net = nn.Sequential(nn.Linear(GAIN_FEATURES, 16), nn.SiLU(), nn.Linear(16, 1))
        if self.use_readout:
            self.degree_out = zero_linear(nn.Linear(DEGREE_FEATURE_DIM, 1))
        if self.use_scaled_gain:
            self.gain_scale = nn.Parameter(torch.tensor(1.0))
            self.gain_net = nn.Sequential(nn.Linear(GAIN_FEATURES, 16), nn.SiLU(),
                                          zero_linear(nn.Linear(16, 1)))

    def degree_bias(self, layer, topology):
        if topology.cached_bias is not None:
            return topology.cached_bias[layer]
        return self.degree_proj[layer](topology.degree_feature).unsqueeze(1)

    def gate_scale(self, layer, topology):
        """Bounded multiplicative modulation, identity when the gate is zero."""
        if topology.cached_scale is not None:
            return topology.cached_scale[layer]
        shift = (self.gate[layer].tanh() * .5 * topology.degree_centered).clamp(-.5, .5)
        return shift.exp().unsqueeze(1).unsqueeze(-1)

    def readout_bias(self, topology):
        if topology.cached_readout is not None:
            return topology.cached_readout
        return self.degree_out(topology.degree_feature)

    def gain(self, x, topology, product):
        if product is None:
            raise ValueError("the gain compensation needs the adjacency product")
        degree = topology.degree.unsqueeze(1)
        selected = 2 * product
        uncovered = (degree - selected).clamp_min(0)
        occupancy = selected / degree.clamp_min(1)
        local = (2 * x - 1) * (self.beta * selected - 1)
        feature = torch.stack((x, selected, uncovered, occupancy, local,
                               degree.expand_as(selected) / 250), dim=-1)
        scale = F.softplus(self.gain_scale) if self.use_scaled_gain else self.gain_gate.tanh()
        return scale * self.gain_net(feature).squeeze(-1)

    def prepare(self, topology):
        """Static per-graph tensors so evaluation pays one broadcast add, not a matmul."""
        bias = tuple(self.degree_proj[layer](topology.degree_feature).unsqueeze(1)
                     for layer in range(len(self.degree_proj))) if self.use_degree else None
        scale = tuple((self.gate[layer].tanh() * .5 * topology.degree_centered).clamp(-.5, .5).exp()
                      .unsqueeze(1).unsqueeze(-1) for layer in range(self.gate.numel())) \
            if self.use_gate else None
        readout = self.degree_out(topology.degree_feature) if self.use_readout else None
        return bias, scale, readout


class StructureGCN(GCN):
    """RLNN backbone with the structural compensations spliced in."""

    def __init__(self, hidden, layers, mode, beta=1.02):
        super().__init__(1, hidden, layers, 1)
        self.mode = mode
        self.compensation = StructureCompensation(hidden, layers, mode, beta)
        self.needs_product = self.compensation.use_gain

    def forward(self, x, edge_index, adj=None, topology=None, adjacency_product=None):
        if topology is None:
            topology = structure_topology(edge_index, adj if adj is not None else edge2adj(edge_index, x.shape[0]),
                                          x.shape[0])
        adj = topology.normalized
        hidden = self.initial_project(x.unsqueeze(-1))
        for layer, (conv, linear) in enumerate(zip(self.convs, self.linears)):
            message = conv(hidden, adj)
            if self.compensation.use_gate:
                message = message * self.compensation.gate_scale(layer, topology)
            hidden = self.activation(message + linear(hidden)) + hidden
            if self.compensation.use_degree:
                hidden = hidden + self.compensation.degree_bias(layer, topology)
        logits = self.final_project(hidden).squeeze(-1)
        if self.compensation.use_readout:
            logits = logits + self.compensation.readout_bias(topology)
        if self.compensation.use_gain:
            logits = logits + self.compensation.gain(x, topology, adjacency_product)
        return logits
