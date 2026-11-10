"""A frozen RLNN backbone with one-shot static two-action priors."""

import torch
from torch import nn

from models import GCN
from v7_structure import FEATURES


MAMBA_MODES = ("v7_mamba", "v7_shuffle_mamba", "v7_degree_mamba", "v7_uni_mamba", "v7_mamba_logit_gate")
CONTEXT_MODES = ("none", "v7_mlp", "v7_message", "v7_static_linear") + MAMBA_MODES


class V7ActionPrior(nn.Module):
    def __init__(self, mode, width=32, d_state=16, feature_dim=len(FEATURES)):
        super().__init__()
        self.mode = mode
        self.order = {"v7_shuffle_mamba": "shuffle", "v7_degree_mamba": "degree"}.get(mode, "bfs") if mode in MAMBA_MODES else None
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))
        self.alpha = nn.Parameter(torch.zeros(()))
        if mode == "v7_static_linear":
            self.out_proj = nn.Linear(feature_dim, 2)
            return
        self.in_proj = nn.Linear(feature_dim, width)
        self.norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, 2)
        if mode in MAMBA_MODES:
            from mamba_ssm import Mamba

            self.sequence = Mamba(d_model=width, d_state=d_state, d_conv=4, expand=2)
            if mode == "v7_mamba_logit_gate":
                self.logit_gate = nn.Parameter(torch.tensor([0., 2.]))
        else:
            expanded = 2 * width
            rank = (width + 15) // 16
            scan_params = expanded * (3 * width + 7 + 2 * rank + 3 * d_state)
            inner = max(1, round((scan_params - width) / (3 * width + 1)))
            self.context = nn.Sequential(nn.Linear(2 * width, inner), nn.SiLU(), nn.Linear(inner, width))

    def normalize(self, x):
        return ((x - self.feature_mean) / self.feature_std).clamp(-5, 5)

    def embeddings(self, topology):
        x = topology.features
        if self.mode == "v7_static_linear":
            return x
        h = self.in_proj(x)
        if self.mode == "v7_mlp":
            context = torch.cat([h[a:b].mean(0, keepdim=True).expand(b-a, -1)
                                 for a, b in zip(topology.ptr, topology.ptr[1:]) if b > a]) if h.numel() else h
            delta = self.context(torch.cat((h, context), dim=-1))
        elif self.mode == "v7_message":
            first = topology.normalized @ h
            delta = self.context(torch.cat((first, topology.normalized @ first), dim=-1))
        else:
            delta = torch.zeros_like(h)
            for order in topology.orders:
                seq = h[order]
                if self.mode == "v7_uni_mamba":
                    encoded = self.sequence(seq)
                else:
                    # Forward and reverse sequences share parameters and occupy independent batch rows.
                    forward, reverse = self.sequence(torch.cat((seq, seq.flip((1,))), dim=0)).chunk(2)
                    encoded = (forward + reverse.flip((1,))) * 0.5
                index = order.flatten()
                delta = delta.index_add(0, index, encoded.flatten(0, 1).to(delta.dtype) / topology.views)
        return self.norm(h + delta)

    def precompute(self, topology):
        return self.out_proj(self.embeddings(topology)).float().tanh()

    def scaled(self, bias, limit):
        return limit * self.alpha.tanh() * bias


class V7StaticContextGCN(GCN):
    def __init__(self, hidden, layers, mode, width, d_state, delta_limit):
        super().__init__(1, hidden, layers, 1)
        self.adapter = V7ActionPrior(mode, width, d_state)
        self.delta_limit = delta_limit

    def forward(self, x, edge_index, _energy_adj, topology):
        base = super().forward(x.unsqueeze(-1), edge_index, topology.normalized).squeeze(-1)
        bias = topology.cached_action_bias
        if bias is None:
            if not self.training:
                raise ValueError("Evaluation requires the one-shot action bias")
            bias = self.adapter.scaled(self.adapter.precompute(topology), self.delta_limit)
        bias = bias.to(base.dtype)
        delta = torch.where(x.bool(), bias[:, 1:2], bias[:, 0:1])
        if self.adapter.mode == "v7_mamba_logit_gate":
            a, b = self.adapter.logit_gate.to(base.dtype).unbind()
            delta = delta * (a * base + b).sigmoid()
        return base + delta
