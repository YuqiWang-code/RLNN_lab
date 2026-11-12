"""v8 ACMD-RLNN: archive-aware correlated macro-transition distillation.

One macro step replaces ``h`` original micro steps.  The student keeps the frozen
backbone's GCN and only widens the read-out from ``1`` to ``2M`` logits: ``M``
components for the continuation state ``Y`` and ``M`` for the block record ``B``.
A single latent component is drawn per (particle, macro step) and shared by all
nodes and by both heads, so marginalising it induces node-node and Y-B
correlation while every conditional distribution stays a product of Bernoullis.

Teacher blocks come from the frozen policy: ``Y = x_h`` and ``B = argmin_{1<=j<=h}
E(x_j)`` (the block's own best state; ``x_0`` is already in the archive, so
including it would teach the record head to copy its input).

Everything here is torch-only and has no dependency on the v7 modules.
"""
from dataclasses import dataclass

import torch
from torch import nn

from models import GCN


def widen_readout(state_dict, m, *, scale=0.02, generator=None):
    """Copy a 1-output backbone into a 2M-output student read-out.

    The first component is an exact copy, so M=1 reproduces the teacher bit for
    bit at initialisation; later components get a small reproducible perturbation
    so the mixture cannot stay symmetric forever (§7.8).
    """
    widened = dict(state_dict)
    weight = state_dict["final_project.weight"]
    bias = state_dict["final_project.bias"]
    rows_w = [weight]
    rows_b = [bias]
    for _ in range(2 * m - 1):
        noise = torch.randn(weight.shape, generator=generator) * scale
        rows_w.append(weight + noise)
        rows_b.append(bias)
    widened["final_project.weight"] = torch.cat(rows_w, dim=0)
    widened["final_project.bias"] = torch.cat(rows_b, dim=0)
    return widened


class MacroKernel(nn.Module):
    """Backbone GCN with a ``2M`` head layout (M continuation + M record)."""

    def __init__(self, components, *, hidden=128, layers=5):
        super().__init__()
        self.components = components
        self.net = GCN(1, hidden, layers, 2 * components)

    def forward(self, state, edge_index, adj):
        """``state`` is ``[N, K]``; returns logits ``[N, K, 2M]``."""
        logits = self.net(state.unsqueeze(-1), edge_index, adj)
        return logits.reshape(state.shape[0], state.shape[1], 2 * self.components)


def sample_component(components, particles, *, generator, device):
    """One latent component per particle, redrawn at every macro step."""
    return torch.randint(0, components, (particles,), generator=generator, device=device)


def sample_macro(kernel, state, edge_index, adj, *, generator, use_record=True):
    """Draw one macro step: ``(Y, B, component)`` from the mixture.

    Tokens are sampled per node conditional on the particle's component, which is
    exactly the product-of-Bernoullis conditional in §7.3.
    """
    logits = kernel(state, edge_index, adj)
    components = kernel.components
    particle = sample_component(components, state.size(1), generator=generator, device=state.device)
    index = torch.arange(state.size(1), device=state.device)
    prob_y = torch.sigmoid(logits[..., 0:components][:, index, particle])
    noise = torch.rand(state.shape, generator=generator, device=state.device, dtype=state.dtype)
    continuation = torch.where(noise < prob_y, 1 - state, state)
    if not use_record:
        return continuation, None, particle
    prob_b = torch.sigmoid(logits[..., components:][:, index, particle])
    noise = torch.rand(state.shape, generator=generator, device=state.device, dtype=state.dtype)
    record = torch.where(noise < prob_b, 1 - state, state)
    return continuation, record, particle


def energy(state, product, bias):
    """MIS energy per column; ``bias`` is the graph's ``b`` vector (all -1)."""
    return (state * (1.02 * product + bias.reshape(-1, 1))).sum(dim=0)


@dataclass
class Block:
    """One teacher block: start state, continuation, block record."""

    state: torch.Tensor      # [N, K] uint8/bool
    continuation: torch.Tensor
    record: torch.Tensor
    record_index: torch.Tensor
    energy_start: torch.Tensor
    energy_end: torch.Tensor
    energy_record: torch.Tensor


@torch.no_grad()
def teacher_blocks(solver, graph, adj, generator, *, horizon, starts, num_t, beta=1.02):
    """Run the frozen teacher once and cut it into blocks at the requested steps.

    ``starts`` are micro-step indices; each block needs ``start + horizon <= num_t``.
    """
    problem = solver.problem
    states = []
    energies = []
    for step, state in enumerate(solver.sample_states(graph, adj, generator=generator,
                                                      normalized=solver.policy_topology(graph, adj))):
        product = adj @ state
        states.append(state.clone())
        energies.append(energy(state, product, graph.b))
    steps = len(states) - 1
    blocks = []
    for start in starts:
        if start + horizon > steps:
            continue
        stacked = torch.stack(energies[start + 1:start + horizon + 1])   # [h, K]
        offset = stacked.argmin(dim=0)                                   # [K], ties -> first
        record_index = offset + 1                                        # micro-step 1..h
        window_states = torch.stack(states[start + 1:start + horizon + 1])   # [h, N, K]
        gather = offset.view(1, 1, -1).expand(1, window_states.size(1), -1)
        record = torch.gather(window_states, 0, gather).squeeze(0)            # [N, K]
        blocks.append(Block(state=states[start], continuation=states[start + horizon],
                            record=record.contiguous(), record_index=record_index,
                            energy_start=energies[start], energy_end=energies[start + horizon],
                            energy_record=stacked.min(dim=0).values))
    return blocks


def joint_log_likelihood(logits, components, state, labels, *, weight_record=1.0):
    """``-log P(Y, B | G, x)`` for a batch of particles, marginal over the mixture.

    ``logits`` is ``[N, K, 2M]``, ``labels`` is ``(Y, B | None)`` with ``[N, K]``.
    The component is marginalised with ``logsumexp`` so that a mixture that only
    matches the marginals is penalised (§7.5); normalising by the node count
    happens outside the mixture.
    """
    y, b = labels
    n, k, _ = logits.shape
    heads = [y] if b is None else [y, b]
    total = 0.0
    for index, label in enumerate(heads):
        block = logits[..., index * components:(index + 1) * components]     # [N, K, M]
        target = label.unsqueeze(-1).to(block.dtype)
        terms = torch.nn.functional.logsigmoid(block) * target + \
                torch.nn.functional.logsigmoid(-block) * (1 - target)
        total = total + terms.sum(dim=0)                                     # [K, M]
        if index == 1:
            total = total * weight_record
    log_pi = -torch.log(torch.tensor(float(components), device=logits.device))
    return -torch.logsumexp(total + log_pi, dim=1).sum()


def marginal_bce(logits, components, labels):
    """Control arm A6: average the components first, then BCE (§7.5 wrong way)."""
    y, b = labels
    total = 0.0
    for index, label in enumerate([y] if b is None else [y, b]):
        block = logits[..., index * components:(index + 1) * components]
        probability = torch.sigmoid(block).mean(dim=-1)
        total = total + torch.nn.functional.binary_cross_entropy(
            probability.clamp(1e-6, 1 - 1e-6), label.to(probability.dtype), reduction="sum")
    return total
