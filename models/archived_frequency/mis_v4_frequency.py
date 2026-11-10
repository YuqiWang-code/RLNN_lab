"""Lazy random-walk state filters with residual adapters for MIS flip logits."""
from typing import NamedTuple

import torch
from torch import nn

from models import GCN, edge2adj


V4_MODES = (
    "v4a_mlp", "v4a_message", "v4a", "v4a_fixed", "v4a_low", "v4a_high",
    "v4a_near", "v4a_norelease", "v4a_noenergy", "v4a_complementary",
)


class V4Topology(NamedTuple):
    normalized: torch.Tensor
    binary: torch.Tensor
    degree: torch.Tensor
    log_degree: torch.Tensor


def v4_topology(edge_index, adj, nodes):
    """Cache static topology, using unit edges instead of the energy's half weights."""
    binary = torch.sparse_csr_tensor(
        adj.crow_indices(), adj.col_indices(),
        torch.ones(adj.values().shape, device=adj.device, dtype=torch.float32),
        size=adj.shape, device=adj.device,
    )
    degree = binary.crow_indices().diff().float().view(-1, 1)
    return V4Topology(edge2adj(edge_index, nodes), binary, degree, degree.log1p())


def neighbor_sum(signals, topology):
    return (topology.binary @ signals.flatten(1)).reshape_as(signals)


def lazy_walk(signals, topology):
    """Apply P = (I + D^-1 A)/2, with the identity on isolated vertices."""
    with torch.autocast(device_type=signals.device.type, enabled=False):
        signals = signals.float()
        degree = topology.degree.reshape((-1,) + (1,) * (signals.ndim - 1))
        average = neighbor_sum(signals, topology) / degree.clamp_min(1)
        return 0.5 * (signals + torch.where(degree > 0, average, signals))


def state_bands(x, topology, beta=1.02):
    """Compute current-state statistics and the exact L2 + B1 + B0 decomposition."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        c = neighbor_sum(x, topology)
        release = ((x == 0) & (c == 1)).float()
        r = x * neighbor_sum(release, topology)
        z = torch.stack((x, c.log1p()), -1)
        pz = lazy_walk(z, topology)
        ppz = lazy_walk(pz, topology)
        return {
            "x": x, "c": c, "r": r, "release": release,
            "z": z, "pz": pz, "ppz": ppz, "L2": ppz,
            "B0": z - pz, "B1": pz - ppz,
            "delta_energy": (1 - 2 * x) * (beta * c - 1),
        }


def encoder(channels, hidden):
    return nn.Sequential(nn.Linear(channels, 16), nn.SiLU(),
                         nn.Linear(16, hidden), nn.LayerNorm(hidden))


class V4FrequencyAdapter(nn.Module):
    """Two equal-width branches with independent state-conditioned sigmoid gates.

    Every arm has the same parameters. The MLP arm uses raw local statistics;
    the message arm uses (P²Z, [Z, PZ]), retaining exactly the frequency arm's
    information. All main arms share eta = mean(abs([B0, B1])) and clipped
    energy changes in [-1, 1] in their six gate inputs.
    """

    def __init__(self, hidden, beta=1.02, mode="v4a"):
        super().__init__()
        if mode not in V4_MODES:
            raise ValueError(f"Unknown v4 frequency mode: {mode}")
        self.beta = beta
        self.mode = mode
        self.low = encoder(2, hidden)
        self.high = encoder(4, hidden)
        self.gates = nn.Linear(6, 2)
        self.alpha_low = nn.Parameter(torch.tensor(0.01))
        self.alpha_high = nn.Parameter(torch.tensor(0.01))
        self.gate_override = None
        self.low_override = None
        self.high_override = None

    def inspect(self, x, adj, topology):
        """Return current filters, gates, encoded branches and injected residuals.

        ``gate_override=0.5`` fixes both gates. Setting ``low_override`` or
        ``high_override`` to False disables that branch during inference.
        ``adj`` is the energy adjacency; all filter reductions use the cached
        binary adjacency in ``topology``.
        """
        details = state_bands(x, topology, self.beta)
        z, pz, ppz = details["z"], details["pz"], details["ppz"]
        b0, b1 = details["B0"], details["B1"]
        high_input = torch.cat((b0, b1), -1)
        low_input = ppz
        eta_input = high_input
        if self.mode == "v4a_mlp":
            low_input = z
            high_input = torch.cat((
                z, topology.log_degree.expand_as(x).unsqueeze(-1),
                details["r"].log1p().unsqueeze(-1),
            ), -1)
        elif self.mode == "v4a_message":
            high_input = torch.cat((z, pz), -1)
        elif self.mode == "v4a_near":
            high_input = torch.cat((b0, torch.zeros_like(b1)), -1)
            eta_input = high_input

        release_input = details["r"].log1p()
        energy_input = details["delta_energy"].clamp(-1, 1)
        if self.mode == "v4a_norelease":
            release_input = torch.zeros_like(release_input)
        if self.mode == "v4a_noenergy":
            energy_input = torch.zeros_like(energy_input)
        eta = eta_input.abs().mean(-1)
        gate_input = torch.stack((
            details["x"], topology.log_degree.expand_as(x),
            details["c"].log1p(), release_input, energy_input, eta,
        ), -1)
        gate_logits = self.gates(gate_input)
        gates = gate_logits.sigmoid()
        if self.mode == "v4a_complementary":
            g_low = (gate_logits[..., 0] - gate_logits[..., 1]).sigmoid()
            gates = torch.stack((g_low, 1 - g_low), -1)
        if self.mode == "v4a_fixed" or self.gate_override is not None:
            value = 0.5 if self.gate_override is None else self.gate_override
            gates = torch.full_like(gates, value)

        u_low, u_high = self.low(low_input), self.high(high_input)
        delta_low = self.alpha_low * gates[..., :1] * u_low
        delta_high = self.alpha_high * gates[..., 1:] * u_high
        low_enabled = self.mode not in ("v4a_high", "v4a_near")
        high_enabled = self.mode != "v4a_low"
        if self.low_override is not None:
            low_enabled = self.low_override
        if self.high_override is not None:
            high_enabled = self.high_override
        if not low_enabled:
            delta_low = torch.zeros_like(delta_low)
        if not high_enabled:
            delta_high = torch.zeros_like(delta_high)
        details.update(
            low_input=low_input, high_input=high_input, eta=eta,
            gate_input=gate_input, gates=gates, u_low=u_low, u_high=u_high,
            delta_low=delta_low, delta_high=delta_high,
        )
        return details

    def forward(self, x, adj, topology):
        details = self.inspect(x, adj, topology)
        return details["delta_low"], details["delta_high"]


class V4FrequencyGCN(GCN):
    def __init__(self, hidden, layers, beta=1.02, mode="v4a"):
        super().__init__(1, hidden, layers, 1)
        self.adapter = V4FrequencyAdapter(hidden, beta, mode)

    def forward(self, x, edge_index, adj, topology=None, *, disable_adapter=False):
        if topology is None:
            topology = v4_topology(edge_index, adj, x.size(0))
        hidden = self.initial_project(x.unsqueeze(-1))
        if not disable_adapter:
            low, high = self.adapter(x, adj, topology)
            hidden = hidden + low.to(hidden.dtype) + high.to(hidden.dtype)
        return self.project_hidden(hidden, topology.normalized).squeeze(-1)
