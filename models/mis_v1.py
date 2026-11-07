"""State-conditioned filters, candidate selection and flip values for MIS."""
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from models import GCN, edge2adj


def adjacency(graph):
    return torch.sparse_coo_tensor(
        graph.edge_index, graph.edge_weight, (graph.num_nodes, graph.num_nodes),
    ).to_sparse_csr()


def state_features(x, adj, beta):
    selected = 2 * (adj @ x)
    degree = adj.crow_indices().diff().to(x.dtype).unsqueeze(1)
    scale = degree.clamp_min(1)
    neighbor_fraction = selected / scale
    gain = (2 * x - 1) * (beta * selected - 1) / scale
    features = torch.stack((x, degree.log1p().expand_as(x), neighbor_fraction, gain), dim=-1)
    return features, torch.where(degree > 0, neighbor_fraction, x)


class MISStateGCN(GCN):
    def __init__(self, hidden, layers, beta, mode):
        super().__init__(4, hidden, layers, 1)
        self.beta = beta
        self.mode = mode
        if mode in {"low", "fixed", "local", "dynamic"}:
            self.low = nn.Linear(1, hidden)
        if mode in {"high", "fixed", "local", "dynamic"}:
            self.high = nn.Linear(1, hidden)
        if mode in {"local", "dynamic"}:
            self.gate = nn.Sequential(nn.Linear(2 if mode == "local" else 4, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, x, edge_index, adj, normalized=None):
        features, neighbor_mean = state_features(x, adj, self.beta)
        hidden = self.initial_project(features)
        if self.mode != "stats":
            low = (0.5 * (x + neighbor_mean)).unsqueeze(-1)
            high = x.unsqueeze(-1) - low
            if self.mode == "low":
                hidden = hidden + F.silu(self.low(low))
            elif self.mode == "high":
                hidden = hidden + F.silu(self.high(high))
            else:
                weight = 0.5
                if self.mode == "local":
                    weight = self.gate(torch.cat((low, high), dim=-1)).sigmoid()
                elif self.mode == "dynamic":
                    weight = self.gate(features).sigmoid()
                hidden = hidden + weight * F.silu(self.low(low)) + (1 - weight) * F.silu(self.high(high))
        if normalized is None:
            normalized = edge2adj(edge_index, x.size(0))
        return self.project_hidden(hidden, normalized).squeeze(-1)


class FlipValue(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))

    @staticmethod
    def features(x, logits, adj, beta):
        state, _ = state_features(x, adj, beta)
        return torch.cat((state, logits.unsqueeze(-1)), dim=-1).float()


def best_decoded(problem, states, adj, bias):
    decoded = problem.decode_gradient_flip(states, adj, bias, return_solutions=True)
    return decoded[:, decoded.sum(dim=0).argmax()]


def search(solver, graph, generator, *, archive="off", probes=5, candidates=4,
           value=None, value_weight=0.5, time_budget=None, diagnostic=False):
    """Keep each trajectory's energy record and optionally compare extra decoded candidates."""
    start = perf_counter()
    adj = adjacency(graph)
    problem = solver.problem
    beta = problem.beta

    def adjust(x, logits):
        features = FlipValue.features(x, logits, adj, beta)
        gain = features[..., 3] if isinstance(value, str) else value(features).squeeze(-1)
        return logits + value_weight * gain.to(logits.dtype)

    states = solver.sample_states(graph, adj, generator=generator, adjust=adjust if value is not None else None)
    best_states = next(states)
    best_energy, _ = problem.energy_func(adj, graph.b, best_states)
    extra_best = None
    extra_count = 0
    extra_seconds = 0.0
    history = [best_states] if diagnostic else None
    probe_steps = {i * solver.num_t // (probes + 1) for i in range(1, probes + 1)} - {0}
    if time_budget is not None:
        incumbent = best_decoded(problem, best_states, adj, graph.b)
        if best_states.is_cuda:
            torch.cuda.synchronize(best_states.device)
        if perf_counter() - start > time_budget:
            incumbent = torch.zeros_like(incumbent)

    for step, state in enumerate(states, 1):
        probe = archive != "off" and step in probe_steps
        if probe and state.is_cuda:
            torch.cuda.synchronize(state.device)
        if probe:
            started = perf_counter()
        energy, grad = problem.energy_func(adj, graph.b, state, compute_grad=probe and archive == "quality")
        improves = energy < best_energy
        best_states = torch.where(improves.unsqueeze(0), state, best_states)
        best_energy = torch.where(improves, energy, best_energy)
        if history is not None:
            history.append(state)

        if probe:
            score = (state.bool() & (grad < 0)).sum(dim=0) if archive == "quality" else -energy
            indices = score.topk(min(candidates, state.size(1))).indices
            solution = best_decoded(problem, state[:, indices], adj, graph.b)
            if extra_best is None:
                extra_best = solution
            else:
                extra_best = torch.where(solution.sum() > extra_best.sum(), solution, extra_best)
            extra_count += len(indices)
            if state.is_cuda:
                torch.cuda.synchronize(state.device)
            extra_seconds += perf_counter() - started

        if time_budget is not None and (step % 5 == 0 or step == solver.num_t or probe):
            current = best_decoded(problem, best_states, adj, graph.b)
            if extra_best is not None:
                current = torch.where(extra_best.sum() > current.sum(), extra_best, current)
            if state.is_cuda:
                torch.cuda.synchronize(state.device)
            if perf_counter() - start > time_budget:
                break
            incumbent = torch.where(current.sum() > incumbent.sum(), current, incumbent)

    details = {"extra_candidates": extra_count, "archive_seconds": extra_seconds}
    if time_budget is not None:
        return incumbent, details
    baseline = best_decoded(problem, best_states, adj, graph.b)
    solution = baseline
    if extra_best is not None:
        solution = torch.where(extra_best.sum() > baseline.sum(), extra_best, baseline)
    details["archive_gain"] = (solution.sum() - baseline.sum()).item()
    if history is not None:
        upper = max(
            best_decoded(problem, torch.cat(history[i:i + 5], dim=1), adj, graph.b).sum().item()
            for i in range(0, len(history), 5)
        )
        details["archive_gap"] = upper - solution.sum().item()
        details["baseline_archive_gap"] = upper - baseline.sum().item()
    return solution, details
