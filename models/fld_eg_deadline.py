"""PyTorch reconstruction of FLD-EG for the NeurIPS 2025 RB-small MIS setup."""
import math

import numpy as np
import torch

from mis_v1 import adjacency
from problems.mis import Problem
from timed_mis_archive import TimedArchive


HORIZON = 300
PARTICLES = 200
ALPHA = 1.2
TAU0 = 0.01
PENALTY = 1.02
GRADIENT_RANK = 5
NOISE_RANK = 9
A_ETA = 4.0
B_ETA = 0.6
C_ALPHA = math.gamma(ALPHA - 1) / math.gamma(ALPHA / 2) ** 2


def greedy_archive(states, neighbors):
    """Greedily take archived 1s first, using ascending node order within ties."""
    packed = np.packbits(states.T.cpu().numpy().astype(bool), axis=1, bitorder="little")
    all_nodes = (1 << len(neighbors)) - 1
    best = []
    for row in packed:
        preferred = int.from_bytes(row, "little")
        remaining = all_nodes
        selected = []
        while remaining:
            candidates = remaining & preferred or remaining
            bit = candidates & -candidates
            node = bit.bit_length() - 1
            selected.append(node)
            remaining &= ~(bit | neighbors[node])
        if len(selected) > len(best):
            best = selected
    return np.asarray(sorted(best), dtype=int)


class TimedFLDEG(TimedArchive):
    def __init__(self):
        self.problem = Problem(PENALTY)
        super().__init__(torch.device("cuda", 0), HORIZON)

    def prepare(self, graph):
        neighbors = [0] * graph.num_nodes
        for source, target in graph.edge_index.T.tolist():
            neighbors[source] |= 1 << target
        return super().prepare(graph), neighbors

    def initialize(self, prepared, generator):
        graph, _ = prepared
        adj = adjacency(graph).to_dense()
        x = torch.randint(0, 2, (graph.num_nodes, PARTICLES), device=self.device,
                          dtype=graph.b.dtype, generator=generator)
        energy, gradient = self.problem.energy_func(adj, graph.b, x, True)
        return adj, (x, x.clone(), gradient, x, energy)

    def advance(self, prepared, adj, state, steps, generator):
        graph, _ = prepared
        x, h, gradient, best, best_energy = state

        def uniform():
            return torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)

        for iteration in range(steps + 1, steps + 11):
            tau = TAU0 * max(self.num_t - iteration, 1) / self.num_t
            eta = (A_ETA / (iteration + 1)) ** B_ETA
            delta = (2 * x - 1) * gradient / 2
            ranked = torch.topk(delta, NOISE_RANK, dim=0).values
            grad_mask = uniform() < torch.sigmoid((delta - ranked[GRADIENT_RANK - 1]) / tau)
            noise_mask = uniform() < torch.sigmoid((-delta - ranked[NOISE_RANK - 1]) / tau)

            angle = math.pi * (uniform() - 0.5)
            tiny = torch.finfo(x.dtype).eps
            exponential = -uniform().clamp_min(tiny).log()
            noise = torch.sin(ALPHA * angle) / torch.cos(angle).clamp_min(tiny).pow(1 / ALPHA)
            noise *= (torch.cos((1 - ALPHA) * angle) / exponential.clamp_min(tiny)).pow((1 - ALPHA) / ALPHA)

            # Algorithm 1 samples a zero with probability h, so descent in x increases h for positive gradients.
            h = (h + grad_mask * (eta * C_ALPHA / tau) * gradient
                 + noise_mask * eta ** (1 / ALPHA) * noise).clamp(0, 1)
            x = (uniform() >= h).to(x.dtype)
            energy, gradient = self.problem.energy_func(adj, graph.b, x, True)
            improves = energy < best_energy
            best = torch.where(improves.unsqueeze(0), x, best)
            best_energy = torch.where(improves, energy, best_energy)
        return x, h, gradient, best, best_energy

    def decode(self, prepared, adj, state):
        _, neighbors = prepared
        return greedy_archive(state[3], neighbors)

    @staticmethod
    def protocol():
        return dict(source="NeurIPS 2025 Algorithm 1 and Table 3 reconstruction", precision="fp32",
                    particles=PARTICLES, horizon=HORIZON, block_steps=10, alpha=ALPHA, tau0=TAU0,
                    penalty=PENALTY, gradient_rank=GRADIENT_RANK, noise_rank=NOISE_RANK,
                    a_eta=A_ETA, b_eta=B_ETA, terminal_temperature_floor=TAU0 / HORIZON,
                    decoding="midpoint and terminal greedy decoding of per-chain energy archives")
