"""Chunked CUDA-graph search for the deployment (deadline) path.

``models/profile_budget_step.py`` measured the frozen step back to back on one
machine: 2.16 ms/step eager against 0.33 ms/step when the same step is captured
as a CUDA graph.  Almost all of the eager time is the forward pass (1.70 ms), so
the gap is launch latency over many small kernels, not arithmetic.  A deadline
run therefore pays for ~123 steps at 0.5 s where the same clock would buy ~490.

This module captures the frozen step in blocks of ``chunk`` steps and replays
them until the deadline.  The block body is the eager body line for line -- same
``flip_logits``, same ``torch.rand`` draw, same ``torch.where`` update, same
archive comparison -- so for a given seed the trajectory is bit-identical.
``models/check_capture_budget.py`` asserts that before any timing is believed.

The decode is *not* captured: ``solution_masks`` runs the A.3 local search with
host-side control flow.  It stays outside, which also matches the protocol
budget: the frozen pipeline decodes every 10 steps and the chunk is 10 steps.

    search = CapturedSearch(solver, graph, adj, topology, generator, chunk=10)
    steps, decode_seconds = search.run(cutoff, decode)   # decode(best_states)
"""
from time import perf_counter

import torch


class CapturedSearch:
    """One graph, one seed, captured in fixed-size blocks and replayable."""

    def __init__(self, solver, graph, adjacency, topology, generator, *, chunk=10):
        if not torch.cuda.is_available():
            raise RuntimeError("captured search requires CUDA")
        self.solver = solver
        self.graph = graph
        self.node_count = graph.num_nodes
        self.chunk = chunk
        self.device = graph.b.device
        self.problem = solver.problem
        self.dtype = solver.dtype
        self.generator = generator
        # Same initial draw as ``sample_states``: the freeze means the archive and
        # the flip policy see exactly this state, so the block starts from x_0.
        self.state = torch.randint(0, 2, (self.node_count, solver.num_k),
                                   device=self.device, dtype=self.dtype, generator=generator)
        self.product = (adjacency @ self.state) if solver.structure_gain else None
        self.best_states = self.state.clone()
        self.best_energy = self.problem.energy_func(adjacency, graph.b, self.state)[0]
        self._adjacency = adjacency
        self._topology = topology
        self._forward = solver.flip_logits
        self._beta = self.problem.beta
        self._b = graph.b
        self._build()

    def _body(self):
        """Advance ``chunk`` frozen steps, updating the archive exactly like the eager loop.

        The order is flip first, then score the new state, which is what the eager
        loop does: (x_0, E(x_0)) is already in the archive from the constructor, and
        each step contributes (x_t, E(x_t)) for t = 1..chunk.  Archiving before the
        flip instead would stop one step short at every block boundary.
        """
        solver, problem = self.solver, self.problem
        for _ in range(self.chunk):
            product = self.product if solver.structure_gain else None
            logits = solver.flip_logits(self.state, self.graph, self._adjacency,
                                        self._topology, adjacency_product=product)
            noise = torch.rand(self.state.shape, device=self.device, dtype=self.dtype,
                               generator=self.generator)
            flipped = torch.where(noise < logits.sigmoid(), 1 - self.state, self.state)
            self.state.copy_(flipped)
            if solver.structure_gain:
                self.product.copy_(self._adjacency @ self.state)
            energy = problem.energy_func(self._adjacency, self._b, self.state)[0]
            improves = energy < self.best_energy
            self.best_states.copy_(torch.where(improves.unsqueeze(0), self.state, self.best_states))
            self.best_energy.copy_(torch.where(improves, energy, self.best_energy))

    def _build(self):
        device = self.device
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        rng_state = self.generator.get_state()
        state, best, energy = self.state.clone(), self.best_states.clone(), self.best_energy.clone()
        with torch.cuda.stream(stream):
            self._body()
        torch.cuda.current_stream(device).wait_stream(stream)
        # Restore the exact pre-capture state: the capture has to record the same
        # draws the eager run would have taken from here.
        self.generator.set_state(rng_state)
        self.state.copy_(state)
        self.best_states.copy_(best)
        self.best_energy.copy_(energy)
        self.graph_exec = torch.cuda.CUDAGraph()
        self.graph_exec.register_generator_state(self.generator)
        with torch.cuda.graph(self.graph_exec, stream=stream):
            self._body()
        torch.cuda.current_stream(device).wait_stream(stream)
        self._warm = 0.0

    def run(self, deadline, decode, *, max_blocks=None):
        """Replay blocks until ``deadline``; returns (steps, decode_seconds, blocks)."""
        steps = 0
        blocks = 0
        decode_seconds = 0.0
        block_time = self._warm
        while True:
            if max_blocks is not None and blocks >= max_blocks:
                break
            if perf_counter() + block_time >= deadline:
                break
            started = perf_counter()
            self.graph_exec.replay()
            steps += self.chunk
            blocks += 1
            decoded = decode(self.best_states)
            decode_seconds += perf_counter() - started
            block_time = perf_counter() - started
        return steps, decode_seconds, blocks
