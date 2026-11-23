"""Deadline scheduler for torch MIS samplers with a decoded archive."""
from time import perf_counter

import torch

from eval_v2 import sync


CLOCK = 0.200


class TimedArchive:
    def __init__(self, device, num_t):
        self.device = device
        self.num_t = num_t
        self.reserve = 0.005

    def prepare(self, graph):
        prepared = graph.clone().to(self.device)
        sync(self.device)
        return prepared

    def warmup(self, graph):
        list(self.solve(graph, 1017, perf_counter(), CLOCK))

    def initialize(self, graph, generator):
        return self.solver.initialize_search(graph, generator)

    def advance(self, graph, adj, state, steps, generator):
        return self.solver.search_steps(graph, adj, state, steps, steps + 10, generator)

    def decode(self, graph, adj, state):
        return self.solver.decode_solution(graph, adj, state[2]).nonzero().flatten().cpu().numpy()

    @torch.no_grad()
    def solve(self, graph, seed, tick, clock):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        adj, state = self.initialize(graph, generator)
        sync(self.device)
        midpoint = (clock + perf_counter() - tick - self.reserve) / 2
        block_time = 0.0
        steps = decoded_steps = 0
        best = -1

        def decode():
            start = perf_counter()
            nodes = self.decode(graph, adj, state)
            finished = perf_counter()
            self.reserve = max(0.005, 2 * (finished - start))
            return finished - tick, nodes

        while steps < self.num_t and perf_counter() - tick + block_time + self.reserve < clock:
            start = perf_counter()
            state = self.advance(graph, adj, state, steps, generator)
            sync(self.device)
            steps += 10
            block_time = perf_counter() - start
            if decoded_steps == 0 and perf_counter() - tick >= midpoint:
                ready, nodes = decode()
                decoded_steps = steps
                if ready <= clock:
                    best = nodes.size
                    yield ready, nodes

        if (steps != decoded_steps or steps == 0) and perf_counter() - tick < clock:
            ready, nodes = decode()
            if ready <= clock and nodes.size > best:
                yield ready, nodes
        self.steps = steps
