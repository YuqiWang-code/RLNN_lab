"""Classical and sampling baselines on the MIS evaluation split."""
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F

from mis_v1 import adjacency, best_decoded


def minimum_degree(graph):
    neighbors = [set() for _ in range(graph.num_nodes)]
    for source, target in graph.edge_index.T.tolist():
        neighbors[source].add(target)
    remaining = set(range(graph.num_nodes))
    solution = torch.zeros(graph.num_nodes, dtype=torch.bool)
    while remaining:
        node = min(remaining, key=lambda i: (len(neighbors[i] & remaining), i))
        solution[node] = True
        remaining.difference_update(neighbors[node] | {node})
    return solution


def kamis(graph, executable, seed, seconds):
    neighbors = [[] for _ in range(graph.num_nodes)]
    for source, target in graph.edge_index.T.tolist():
        neighbors[source].append(str(target + 1))
    with TemporaryDirectory() as folder:
        path = Path(folder) / "input.graph"
        out = Path(folder) / "solution"
        lines = [f"{graph.num_nodes} {graph.edge_index.size(1) // 2}"]
        path.write_text("\n".join(lines + [" ".join(row) for row in neighbors]) + "\n", encoding="ascii")
        subprocess.run([str(executable), str(path), f"--output={out}", f"--seed={seed}",
                        f"--time_limit={seconds}", "--console_log"], check=True, stdout=subprocess.DEVNULL)
        values = [int(value) for value in out.read_text().split()]
    if len(values) != graph.num_nodes or any(value not in (0, 1) for value in values):
        raise ValueError("unexpected ReduMIS solution format")
    return torch.tensor(values, dtype=torch.bool)


def gurobi(graph, env, seed, seconds):
    import gurobipy as gp

    with gp.Model(env=env) as model:
        x = model.addVars(graph.num_nodes, vtype=gp.GRB.BINARY)
        model.setObjective(x.sum(), gp.GRB.MAXIMIZE)
        model.addConstrs(x[u] + x[v] <= 1 for u, v in graph.edge_index.T.tolist() if u < v)
        model.setParam("Threads", 1)
        model.setParam("Seed", seed)
        model.setParam("TimeLimit", seconds)
        model.optimize()
        return torch.tensor([x[i].X > 0.5 for i in range(graph.num_nodes)]) if model.SolCount else torch.zeros(graph.num_nodes, dtype=torch.bool)


def ordered_log_prob(logits, order):
    remaining = logits.clone()
    total = logits.new_zeros(logits.size(1))
    for index in order.unbind(0):
        row = index.unsqueeze(0)
        total += remaining.gather(0, row).squeeze(0) - remaining.logsumexp(dim=0)
        remaining.scatter_(0, row, -torch.inf)
    return total


def isco(problem, graph, generator, steps, particles, radius, temperature):
    """Binary path-auxiliary Metropolis updates with a fixed proposal radius."""
    adj = adjacency(graph)
    x = torch.randint(0, 2, (graph.num_nodes, particles), dtype=graph.b.dtype,
                      device=graph.b.device, generator=generator)
    energy, gradient = problem.energy_func(adj, graph.b, x, True)
    best, best_energy = x, energy
    for step in range(steps):
        temp = temperature * (0.01 ** (step / max(1, steps - 1)))
        logits = F.logsigmoid((2 * x - 1) * gradient / temp)
        noise = torch.empty_like(logits).exponential_(generator=generator).log()
        order = (logits - noise).topk(min(radius, graph.num_nodes), dim=0).indices
        proposed = x.clone()
        proposed.scatter_(0, order, 1 - x.gather(0, order))
        energy_y, grad_y = problem.energy_func(adj, graph.b, proposed, True)
        reverse = F.logsigmoid((2 * proposed - 1) * grad_y / temp)
        log_accept = (energy - energy_y) / temp + ordered_log_prob(reverse, order.flip(0)) - ordered_log_prob(logits, order)
        accept = torch.rand(particles, device=x.device, generator=generator).log() < log_accept
        x = torch.where(accept, proposed, x)
        gradient = torch.where(accept, grad_y, gradient)
        energy = torch.where(accept, energy_y, energy)
        improves = energy < best_energy
        best, best_energy = torch.where(improves, x, best), torch.minimum(best_energy, energy)
    return best_decoded(problem, best, adj, graph.b)
