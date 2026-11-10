"""DAFR-Lite: post-encoding graph bands for scalar MIS flip corrections."""
import torch
from torch import nn

from models import GCN
from archived_frequency.mis_v4_frequency import lazy_walk, v4_topology as v5_topology


V5_MODES = ("v5a", "v5a_mlp", "v5a_message", "v5a_low", "v5a_high", "v5a_fixed",
            "v5a_noenergy", "v5a_conflict", "v5a_mlp_conflict", "v5a_message_conflict")


def local_energy(x, counts, beta):
    return (1 - 2 * x) * (beta * counts - 1)


class V5DAFR(nn.Module):
    def __init__(self, mode="v5a", width=12, beta=1.02):
        super().__init__()
        self.mode = mode
        self.beta = beta
        self.signal = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, 2), nn.Tanh())
        self.gate = nn.Linear(5, 2)
        self.theta_low = nn.Parameter(torch.zeros(()))
        self.theta_high = nn.Parameter(torch.zeros(()))

    def features(self, x, counts, topology):
        degree = topology.degree.expand_as(x)
        delta_e = local_energy(x, counts, self.beta)
        occupancy = counts / degree.clamp_min(1)
        if self.mode.endswith("_conflict"):
            occupancy = ((x == 1) & (counts > 0)).float()
        energy = (delta_e / 5).tanh() if self.mode != "v5a_noenergy" else torch.zeros_like(x)
        return torch.stack((x, counts.log1p(), topology.log_degree.expand_as(x), occupancy, energy), -1)

    def encode(self, z):
        proposal = self.signal(z)
        gates = self.gate(z).sigmoid() if self.mode != "v5a_fixed" else torch.full_like(proposal, 0.5)
        return proposal, gates, proposal * gates

    def bands(self, gated, topology):
        if self.mode.startswith("v5a_mlp"):
            return gated[..., 0], gated[..., 1]
        propagated = lazy_walk(gated, topology)
        low = propagated[..., 0]
        high = (0.5 * (gated[..., 1] + propagated[..., 1]) if self.mode.startswith("v5a_message")
                else gated[..., 1] - propagated[..., 1])
        if self.mode == "v5a_low":
            high = torch.zeros_like(high)
        if self.mode == "v5a_high":
            low = torch.zeros_like(low)
        return low, high

    def forward(self, x, topology, return_details=False):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            counts = topology.binary @ x
            z = self.features(x, counts, topology)
            proposal, gates, gated = self.encode(z)
            low, high = self.bands(gated, topology)
            a_low = 0.25 * self.theta_low.tanh()
            a_high = 0.25 * self.theta_high.tanh()
            correction = a_low * low + a_high * high
        if return_details:
            return correction, dict(z=z, counts=counts, proposal=proposal, gates=gates, gated=gated,
                                    low=low, high=high, a_low=a_low, a_high=a_high)
        return correction


class V5DAFRGCN(GCN):
    def __init__(self, hidden, layers, beta=1.02, mode="v5a"):
        super().__init__(1, hidden, layers, 1)
        self.adapter = V5DAFR(mode=mode, beta=beta)

    def backbone_logits(self, x, edge_index, topology):
        return super().forward(x.unsqueeze(-1), edge_index, topology.normalized).squeeze(-1)

    def forward(self, x, edge_index, energy_adj, topology=None):
        if topology is None:
            topology = v5_topology(edge_index, energy_adj, x.size(0))
        base = self.backbone_logits(x, edge_index, topology)
        return base + self.adapter(x, topology).to(base.dtype)
