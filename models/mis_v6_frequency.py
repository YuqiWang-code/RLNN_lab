"""SC-RLNN with gated backbone bands and a cross-layer readout."""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from models import edge2adj


V6_MODES = ("v6_full", "v6_noscatter", "v6_lowonly", "v6_highonly", "v6_nogate", "v6_noanti")


class V6Topology(NamedTuple):
    normalized: torch.Tensor
    inverse_degree: torch.Tensor
    log_degree: torch.Tensor
    degree_asymmetry: torch.Tensor
    polarization_edges: torch.Tensor
    polarization_weight: torch.Tensor


def physics_state(x, product, inverse_degree, log_degree, asymmetry, beta, use_asymmetry):
    counts = 2 * product.to(x.dtype)
    occupancy = counts * inverse_degree.to(x.dtype)
    delta = (1 - 2 * x) * (beta * counts - 1)
    asymmetry = asymmetry.to(x.dtype).expand_as(x) if use_asymmetry else torch.zeros_like(x)
    return torch.stack((x, occupancy, (0.25 * delta).tanh(),
                        log_degree.to(x.dtype).expand_as(x), asymmetry), dim=-1), counts


def mix_bands(h, local, neighbor, gate, alpha_low, alpha_high, scale, bias, mode):
    if mode == "v6_lowonly":
        mixed = (local + neighbor) * (0.5 + 0.5 * alpha_low.tanh()).to(h.dtype)
    elif mode == "v6_highonly":
        high = (0.5 + 0.5 * alpha_high.tanh()).to(h.dtype) * gate
        mixed = (local - neighbor) * high
    else:
        low = (0.5 + 0.5 * alpha_low.tanh()).to(h.dtype)
        high = (0.5 + 0.5 * alpha_high.tanh()).to(h.dtype) * gate
        mixed = local * (low + high) + neighbor * (low - high)
    return torch.addcmul(h, scale.to(h.dtype), F.silu(mixed + bias.to(h.dtype)))


@torch.no_grad()
def v6_topology(edge_index, energy_adj, num_nodes):
    degree = energy_adj.crow_indices().diff().float()
    inverse_degree = degree.clamp_min(1).reciprocal()
    log_degree = degree.log1p()
    log_degree = log_degree / log_degree.max().clamp_min(1)
    # MIS energy adjacency stores half the undirected adjacency weight.
    neighbor_log = 2 * (energy_adj @ log_degree.to(energy_adj.dtype)[:, None]).flatten()
    src, dst = edge_index
    edges = edge_index[:, src < dst]
    weight = (inverse_degree[edges[0]] * inverse_degree[edges[1]]).sqrt()
    return V6Topology(edge2adj(edge_index, num_nodes), inverse_degree[:, None],
                      log_degree[:, None], (log_degree - inverse_degree * neighbor_log)[:, None],
                      edges, weight[:, None])


class SpectralDecoupledConv(nn.Module):
    def __init__(self, hidden, mode):
        super().__init__()
        self.mode = mode
        self.neighbor = nn.Linear(hidden, hidden, bias=False)
        self.local = nn.Linear(hidden, hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.alpha_low = nn.Parameter(torch.empty(hidden), requires_grad=mode != "v6_highonly")
        self.alpha_high = nn.Parameter(torch.empty(hidden), requires_grad=mode != "v6_lowonly")
        self.res_scale = nn.Parameter(torch.empty(hidden))
        self.mix = mix_bands

    def reset_parameters(self):
        self.neighbor.reset_parameters()
        nn.init.orthogonal_(self.local.weight, gain=0.5)
        nn.init.zeros_(self.bias)
        # Different nonzero gains give the neighbor projection a gradient at initialization.
        nn.init.constant_(self.alpha_low, 0.10)
        nn.init.constant_(self.alpha_high, -0.10)
        nn.init.constant_(self.res_scale, 0.2)

    def forward(self, h, adj, gate):
        support = self.neighbor(h)
        neighbor = torch.spmm(adj, support.flatten(1)).reshape_as(h).to(h.dtype)
        local = self.local(h)
        return self.mix(h, local, neighbor, gate, self.alpha_low, self.alpha_high,
                        self.res_scale, self.bias, self.mode)


class V6SCGCN(nn.Module):
    def __init__(self, hidden, layers, beta, mode):
        super().__init__()
        if layers < 3:
            raise ValueError("v6 requires at least three layers for distinct readout depths")
        self.mode, self.beta = mode, beta
        self.mid = layers // 2
        self.physics = physics_state
        self.input_proj = nn.Linear(5, hidden)
        self.conflict_gate = None if mode in ("v6_nogate", "v6_lowonly") else nn.Linear(5, 1)
        self.convs = nn.ModuleList(SpectralDecoupledConv(hidden, mode) for _ in range(layers))
        self.head_low = nn.Linear(hidden, 1, bias=False)
        self.head_phys = nn.Linear(5, 1)
        self.gamma_phys = nn.Parameter(torch.tensor(0.1))
        if mode != "v6_noscatter":
            self.head_meso = nn.Linear(hidden, 1, bias=False)
            self.head_high = nn.Linear(hidden, 1, bias=False)
            self.gamma_meso = nn.Parameter(torch.tensor(0.2))
            self.gamma_high = nn.Parameter(torch.tensor(0.2))

    def compile_pointwise(self):
        self.physics = torch.compile(physics_state, dynamic=True)
        mix = torch.compile(mix_bands, dynamic=True)
        for conv in self.convs:
            conv.mix = mix

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        if self.conflict_gate is not None:
            nn.init.zeros_(self.conflict_gate.weight)
            nn.init.zeros_(self.conflict_gate.bias)
        for conv in self.convs:
            conv.reset_parameters()
        nn.init.xavier_uniform_(self.head_low.weight)
        nn.init.zeros_(self.head_phys.weight)
        nn.init.zeros_(self.head_phys.bias)
        nn.init.constant_(self.gamma_phys, 0.1)
        if self.mode != "v6_noscatter":
            nn.init.orthogonal_(self.head_meso.weight, gain=0.2)
            nn.init.orthogonal_(self.head_high.weight, gain=0.2)
            nn.init.constant_(self.gamma_meso, 0.2)
            nn.init.constant_(self.gamma_high, 0.2)

    def forward(self, x, edge_index, energy_adj, topology=None, *,
                adjacency_product=None, regularize=False, diagnose=False):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        if topology is None:
            topology = v6_topology(edge_index, energy_adj, x.size(0))
        if adjacency_product is None:
            adjacency_product = energy_adj @ x
        state, counts = self.physics(x, adjacency_product, topology.inverse_degree, topology.log_degree,
                                     topology.degree_asymmetry, self.beta, self.mode != "v6_noanti")
        gate = 1.0 if self.conflict_gate is None else 0.5 + self.conflict_gate(state).sigmoid()
        h = F.silu(self.input_proj(state))
        trace = []
        for layer, conv in enumerate(self.convs, 1):
            h = conv(h, topology.normalized, gate)
            if layer == 1:
                first = h
            if layer == self.mid:
                mid = h
            if diagnose:
                trace.append(h)
        logits = self.head_low(h).squeeze(-1)
        logits = logits + self.gamma_phys.to(logits.dtype) * self.head_phys(state).squeeze(-1)
        high = meso_logit = high_logit = None
        if self.mode != "v6_noscatter":
            high = first - mid
            meso_logit = self.gamma_meso.to(logits.dtype) * self.head_meso(mid - h).squeeze(-1)
            high_logit = self.gamma_high.to(logits.dtype) * self.head_high(high).squeeze(-1)
            logits = logits + meso_logit + high_logit
        logits = logits.reshape(shape)
        if diagnose:
            return logits, trace, meso_logit, high_logit
        if regularize:
            orthogonality = (F.cosine_similarity(h.float(), high.float(), dim=-1).square()
                             if high is not None else None)
            return logits, counts, orthogonality
        return logits


def polarization_bonus(x, prob, counts, topology, num_graphs):
    src, dst = topology.polarization_edges
    selected = (1 - prob) * x + prob * (1 - x)
    plateau = (counts[src] <= 1) & (counts[dst] <= 1)
    difference = (selected[src] - selected[dst]).square()
    return (difference * plateau * topology.polarization_weight).sum(dim=0).mean() / num_graphs
