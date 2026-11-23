"""Padded RLNN search with reusable graph buffers and optional CUDA Graph replay."""

import numpy as np
import torch
from time import perf_counter


class PaddedSearch:
    """One captured block, reused for every graph that fits in ``n_max`` nodes."""

    def __init__(self, solver, *, n_max, capacity, chunk=10, device=None, decode_chunk=4,
                 dense=False, capture=True):
        if not torch.cuda.is_available():
            raise RuntimeError("padded search requires CUDA")
        if solver.frequency != "none" or solver.structure or solver.v7_context:
            raise ValueError("the padded engine mirrors the plain frozen step only")
        self.solver = solver
        self.model = solver.model
        self.scale_operator = getattr(solver, "scale_operator", False)
        self.problem = solver.problem
        self.dtype = solver.dtype
        self.device = device or solver.accelerator.device
        self.n_max = int(n_max)
        self.capacity = int(capacity)
        self.chunk = int(chunk)
        self.num_k = int(solver.num_k)
        self.beta = self.problem.beta
        rows, cap = self.n_max, self.capacity
        self.rows = torch.arange(rows, dtype=torch.int64, device=self.device)
        self.slots = torch.repeat_interleave(self.rows, cap)
        self.crow = torch.arange(0, rows * cap + 1, cap, dtype=torch.int64, device=self.device)
        self.col = torch.zeros(rows * cap, dtype=torch.int64, device=self.device)
        self.raw = torch.zeros(rows * cap, dtype=self.dtype, device=self.device)
        self.norm = torch.zeros(rows * cap, dtype=self.dtype, device=self.device)
        self.real = torch.zeros(rows * cap, dtype=self.dtype, device=self.device)
        self.bias = torch.zeros(rows, 1, dtype=self.dtype, device=self.device)
        self.scale = torch.ones(rows, 1, dtype=self.dtype, device=self.device)
        self.adjacency = torch.sparse_csr_tensor(self.crow, self.col, self.raw, (rows, rows),
                                                 device=self.device)
        self.normalized = torch.sparse_csr_tensor(self.crow, self.col, self.norm, (rows, rows),
                                                  device=self.device)
        self.use_dense = bool(dense)
        self.dense_norm = torch.zeros(rows, rows, dtype=self.dtype, device=self.device)
        self.state = torch.zeros(rows, self.num_k, dtype=self.dtype, device=self.device)
        self.best_states = torch.zeros_like(self.state)
        self.best_energy = torch.zeros(self.num_k, dtype=self.dtype, device=self.device)
        self.improvements = torch.zeros(1, dtype=torch.long, device=self.device)
        self.nodes = 0
        self.temperature = torch.ones((), dtype=self.dtype, device=self.device)
        self.keep = torch.ones(rows, self.num_k, dtype=self.dtype, device=self.device)
        self.tabu = torch.zeros(rows, self.num_k, dtype=torch.bool, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.decoder = None
        if decode_chunk:
            from mis_v8_decode import CapturedDecoder
            template = torch.zeros(rows, self.num_k, dtype=self.dtype, device=self.device)
            self.decoder = CapturedDecoder(self.problem, self.adjacency, self.bias, template,
                                           chunk=decode_chunk)
        if capture:
            self._build()
            self.step_block = self.graph_exec.replay
        else:
            self.step_block = self._body

    def load(self, graph, edges, seed):
        """Load one graph and reset the per-particle archive and RNG."""
        started = perf_counter()
        n = int(graph.num_nodes)
        if n > self.n_max:
            raise ValueError(f"graph has {n} nodes, engine was built for {self.n_max}")
        edges = edges.cpu().numpy()
        src, dst = edges[0], edges[1]
        rows, cap, n_max = self.n_max, self.capacity, self.n_max
        dense = np.zeros((n_max, n_max), dtype=np.uint8)
        dense[src, dst] = 1
        order = np.argsort(dense == 0, axis=1, kind="stable")[:, : cap - 1]
        count = dense.sum(axis=1)
        valid = np.arange(cap - 1)[None, :] < count[:, None]
        cols = np.concatenate([np.arange(n_max)[:, None], np.where(valid, order, n_max)], axis=1)
        mask = np.concatenate([np.ones((n_max, 1), dtype=bool), valid], axis=1)
        rank = np.argsort(cols, axis=1, kind="stable")
        cols = np.take_along_axis(cols, rank, axis=1)
        mask = np.take_along_axis(mask, rank, axis=1)
        index = np.arange(n_max)[:, None]
        self_loop = mask & (cols == index) & (index < n)
        edge = mask & ~self_loop & (cols < n)
        storage = np.where(cols >= n_max, 0, cols)
        self.col.copy_(torch.from_numpy(storage.reshape(-1)))
        self.raw.copy_(torch.from_numpy(edge.astype(np.float32).reshape(-1)))
        self.real.copy_(torch.from_numpy((edge | self_loop).astype(np.float32).reshape(-1)))
        degree = self.raw.reshape(n_max, cap).sum(dim=1)
        inv = (degree + 1).pow(-0.5)
        self.scale.copy_(inv.reciprocal().unsqueeze(1))
        self.norm.copy_((inv[self.slots] * inv[self.col]).mul_(self.real))
        if self.use_dense:
            nonzero = self.norm != 0
            self.dense_norm.zero_()
            self.dense_norm.index_put_((self.slots[nonzero], self.col[nonzero]),
                                       self.norm[nonzero])
        self.bias.zero_()
        self.bias[:n].copy_(graph.b.reshape(-1, 1).to(self.dtype))
        self.nodes = n
        self.generator.manual_seed(int(seed))
        self.state.copy_(torch.randint(0, 2, (n_max, self.num_k), device=self.device,
                                       dtype=self.dtype, generator=self.generator))
        self.best_states.copy_(self.state)
        self.best_energy.copy_(self.problem.energy_func(self.adjacency, self.bias, self.state)[0])
        self.improvements.zero_()
        self.keep.fill_(1.0)
        self.tabu.zero_()
        return perf_counter() - started

    def prime(self, mask, n=None, particle=None):
        """Start particles from ``mask`` (an int bitset over the real nodes).

        Used for initialisation studies (e.g. a greedy start) and for feeding a
        platform result back into the captured search.  ``particle=None`` primes
        every column and resets the archive; a single column is injected without
        touching the archive, so the other particles keep their own basins.
        """
        n = int(self.nodes if n is None else n)
        bits = torch.tensor([(mask >> i) & 1 for i in range(n)], dtype=self.dtype,
                            device=self.device)
        columns = slice(None) if particle is None else slice(particle, particle + 1)
        self.state[:, columns].zero_()
        self.state[:n, columns] = bits.unsqueeze(1).expand(n, self.num_k if particle is None else 1)
        if particle is None:
            self.best_states.copy_(self.state)
            self.best_energy.copy_(self.problem.energy_func(self.adjacency, self.bias, self.state)[0])
            self.improvements.zero_()

    def _logits(self):
        """Frozen policy logits for the current state.

        The plain GCN ignores the un-normaliser; ``--arch scale`` needs it to turn
        the normalised aggregation back into raw hop counts.
        """
        state = self.state.unsqueeze(-1)
        weight = self.dense_norm if self.use_dense else self.normalized
        if self.scale_operator:
            logits = self.model(state, None, weight, self.scale)
        else:
            logits = self.model(state, None, weight)
        return logits.squeeze(-1).masked_fill(self.tabu, torch.finfo(self.dtype).min)

    def evict(self, rows):
        """Empty ``rows`` in every particle and forbid them until ``release``."""
        if not rows:
            return
        index = torch.as_tensor(sorted({int(row) for row in rows}),
                                device=self.device)
        self.keep[index] = 0.0
        self.tabu[index] = True
        self.state[index] = 0.0

    def release(self):
        """Lift every tabu and let the policy use those rows again."""
        self.keep.fill_(1.0)
        self.tabu.zero_()

    def _body(self):
        for _ in range(self.chunk):
            logits = self._logits()
            probs = torch.div(logits, self.temperature).sigmoid()
            noise = torch.rand(self.state.shape, device=self.device, dtype=self.dtype,
                               generator=self.generator)
            flipped = torch.where(noise < probs, 1 - self.state, self.state)
            flipped = flipped * self.keep
            self.state.copy_(flipped)
            energy = self.problem.energy_func(self.adjacency, self.bias, self.state)[0]
            improves = energy < self.best_energy
            self.best_states.copy_(torch.where(improves.unsqueeze(0), self.state, self.best_states))
            self.best_energy.copy_(torch.where(improves, energy, self.best_energy))
            torch.add(self.improvements, improves.sum().reshape(1), out=self.improvements)

    def _build(self):
        device = self.device
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        state, best, energy = self.state.clone(), self.best_states.clone(), self.best_energy.clone()
        improvements = self.improvements.clone()
        with torch.cuda.stream(stream):
            self._body()
        torch.cuda.current_stream(device).wait_stream(stream)
        self.state.copy_(state)
        self.best_states.copy_(best)
        self.best_energy.copy_(energy)
        self.improvements.copy_(improvements)
        self.graph_exec = torch.cuda.CUDAGraph()
        self.graph_exec.register_generator_state(self.generator)
        with torch.cuda.graph(self.graph_exec, stream=stream):
            self._body()
        torch.cuda.current_stream(device).wait_stream(stream)

    def run(self, deadline, decode, *, max_blocks=None, skip_unchanged=False,
            decode_every=1, temperature=1.0, anneal=False, sync_each_block=True,
            on_block=None, decode_reserve=None, midpoint_decode=True):
        steps = blocks = 0
        decode_seconds = 0.0
        block_time = 0.0
        self.improvements.zero_()
        decoded_at = -1
        pending = False
        every = max(1, int(decode_every))
        self.temperature.fill_(float(temperature))
        started = perf_counter()
        window = max(deadline - started, 1e-6)
        midpoint = (started + window / 2
                    if decode_reserve is not None and midpoint_decode else float("inf"))
        while True:
            if anneal and temperature > 1.0:
                left = max(deadline - perf_counter(), 0.0)
                self.temperature.fill_(1.0 + (float(temperature) - 1.0) * left / window)
            if max_blocks is not None and blocks >= max_blocks:
                break
            if perf_counter() + block_time + (decode_reserve or 0.0) >= deadline:
                break
            started = perf_counter()
            self.step_block()
            if sync_each_block:
                torch.cuda.synchronize(self.device)
            steps += self.chunk
            blocks += 1
            if on_block is not None:
                on_block(blocks)
            if skip_unchanged:
                moved = int(self.improvements.item())
                if moved == decoded_at and not (pending and perf_counter() >= midpoint):
                    block_time = perf_counter() - started
                    continue
                decoded_at = moved
            block_finished = perf_counter()
            if blocks % every and block_finished < midpoint:
                pending = True
                block_time = block_finished - started
                continue
            pending = False
            midpoint = float("inf")
            decode(self.best_states)
            decode_elapsed = perf_counter() - block_finished
            decode_seconds += decode_elapsed
            block_time = block_finished - started
            if decode_reserve is not None:
                decode_reserve = max(decode_reserve, 2 * decode_elapsed)
            else:
                block_time += decode_elapsed
        if (pending or (decode_reserve is not None and blocks == 0)) and (
            decode_reserve is None or perf_counter() < deadline
        ):
            started = perf_counter()
            decode(self.best_states)
            decode_seconds += perf_counter() - started
        return steps, decode_seconds, blocks

    def decode_best(self, states, *, count_nodes=None):
        """Best particle of the A.3 pass, packed over the real nodes only."""
        solutions = self.decoder.decode(states)
        n = self.nodes if count_nodes is None else int(count_nodes)
        sizes = solutions[:n].sum(dim=0)
        _, index = torch.max(sizes, dim=0)
        column = solutions[:n, int(index)].to(torch.uint8).cpu().numpy()
        return int.from_bytes(np.packbits(column, bitorder="little").tobytes(), "little")

    def eager_step(self, steps):
        """Reference path: the same padded buffers, no capture (for the gate)."""
        state, best, energy = self.state.clone(), self.best_states.clone(), self.best_energy.clone()
        improvements = self.improvements.clone()
        saved = self.generator.get_state()
        for _ in range(steps):
            self._body_one()
        result = (self.state.clone(), self.best_states.clone(), self.best_energy.clone())
        self.state.copy_(state)
        self.best_states.copy_(best)
        self.best_energy.copy_(energy)
        self.improvements.copy_(improvements)
        self.generator.set_state(saved)
        return result

    def _body_one(self):
        chunk, self.chunk = self.chunk, 1
        try:
            self._body()
        finally:
            self.chunk = chunk
