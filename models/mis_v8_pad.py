"""Amortised capture: one CUDA graph for a whole dataset.

``models/mis_v8_capture.py`` captures the frozen step per graph and buys ~7x
steps, but the capture costs 46-70 ms and it is charged inside the clock.  On the
paper's clock that is the whole budget: at 0.2 s/graph the prelude is 47 % of the
clock and at 0.1 s the run takes *zero* steps (``outputs/v8_macro/author_short``).

This module removes the per-graph capture.  Every graph is padded to ``n_max``
nodes with a fixed per-row capacity, so the CSR layout -- and therefore every
tensor shape and pointer the CUDA graph recorded -- never changes.  One capture
serves the whole dataset; loading a graph is five buffer copies.

The padding is exact rather than approximate:

* a padded node has bias 0 and no incident edge, and the MIS energy is
  ``sum(x * (beta * A x + b))``, so padded nodes contribute exactly 0;
* a padded node's row is ``col = self, weight = 0``, so its normalised weights
  are zero and real rows are untouched;
* ``torch.rand`` and ``torch.randint`` are filled in row-major order for a fixed
  ``num_k``, so ``rand((n_max, k))[:n]`` equals ``rand((n, k))`` drawn from the
  same seed.  The padded trajectory of the real rows is therefore bit-identical
  to the unpadded one, which ``models/check_padded_capture.py`` asserts.

The deployed backbone is a plain ``GCN(1, hidden, layers, 1)`` -- ``state_features``
belongs to the frequency adapters, which this checkpoint does not use -- so its only
graph input is the normalised adjacency.  Padding therefore needs no extra
bookkeeping: zero-weight entries and a zero bias make every padded contribution
exactly zero.

    engine = PaddedSearch(solver, n_max=300, capacity=33)
    engine.load(graph, graph.edge_index, seed)     # outside the clock
    steps, decode_seconds, _ = engine.run(deadline, decode)
"""
import numpy as np
import torch
from time import perf_counter


class PaddedSearch:
    """One captured block, reused for every graph that fits in ``n_max`` nodes."""

    def __init__(self, solver, *, n_max, capacity, chunk=10, device=None, decode_chunk=4):
        if not torch.cuda.is_available():
            raise RuntimeError("padded search requires CUDA")
        if solver.frequency != "none" or solver.structure or solver.v7_context:
            raise ValueError("the padded engine mirrors the plain frozen step only")
        self.solver = solver
        self.model = solver.model
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
        self.adjacency = torch.sparse_csr_tensor(self.crow, self.col, self.raw, (rows, rows),
                                                 device=self.device)
        self.normalized = torch.sparse_csr_tensor(self.crow, self.col, self.norm, (rows, rows),
                                                  device=self.device)
        self.state = torch.zeros(rows, self.num_k, dtype=self.dtype, device=self.device)
        self.best_states = torch.zeros_like(self.state)
        self.best_energy = torch.zeros(self.num_k, dtype=self.dtype, device=self.device)
        self.improvements = torch.zeros(1, dtype=torch.long, device=self.device)
        self.nodes = 0
        # The generator has to be one stable object: the capture registers it.
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.decoder = None
        if decode_chunk:
            from mis_v8_decode import CapturedDecoder
            template = torch.zeros(rows, self.num_k, dtype=self.dtype, device=self.device)
            self.decoder = CapturedDecoder(self.problem, self.adjacency, self.bias, template,
                                           chunk=decode_chunk)
        self._build()

    # ---------------------------------------------------------------- loading
    def load(self, graph, edges, seed):
        """Write one graph into the fixed buffers and reset the archive.

        Runs before the clock starts, exactly like reading a graph file does for
        the external solvers.  The row order matters: the real neighbours and the
        self loop are stored in ascending column order with the padding after
        them, because ``spmm`` accumulates in storage order and a different order
        would round differently from the unpadded CSR.
        """
        started = perf_counter()
        n = int(graph.num_nodes)
        if n > self.n_max:
            raise ValueError(f"graph has {n} nodes, engine was built for {self.n_max}")
        edges = edges.cpu().numpy()
        src, dst = edges[0], edges[1]
        rows, cap, n_max = self.n_max, self.capacity, self.n_max
        dense = np.zeros((n_max, n_max), dtype=np.uint8)
        dense[src, dst] = 1
        # Neighbours of a row come first in ``order`` (ascending column, stable).
        # The self loop is added as column ``r``; the unused slots carry a
        # sentinel that sorts after every real column, so the row ends up in
        # ascending column order with the padding at the end -- the same order the
        # unpadded CSR has, which matters because ``spmm`` accumulates in storage
        # order and a different order would round differently.
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
        # Normalised weights with the same expression as ``edge2adj``:
        # inv[row] * 1 * inv[col], inv = (degree + self loop) ** -0.5, on device in
        # the model dtype so the rounding matches the unpadded path bit for bit.
        degree = self.raw.reshape(n_max, cap).sum(dim=1)
        inv = (degree + 1).pow(-0.5)
        self.norm.copy_((inv[self.slots] * inv[self.col]).mul_(self.real))
        self.bias.zero_()
        self.bias[:n].copy_(graph.b.reshape(-1, 1).to(self.dtype))
        self.nodes = n
        self.generator.manual_seed(int(seed))
        self.state.copy_(torch.randint(0, 2, (n_max, self.num_k), device=self.device,
                                       dtype=self.dtype, generator=self.generator))
        self.best_states.copy_(self.state)
        self.best_energy.copy_(self.problem.energy_func(self.adjacency, self.bias, self.state)[0])
        self.improvements.zero_()
        return perf_counter() - started

    # ------------------------------------------------------------------ step
    def _body(self):
        for _ in range(self.chunk):
            logits = self.model(self.state.unsqueeze(-1), None, self.normalized).squeeze(-1)
            noise = torch.rand(self.state.shape, device=self.device, dtype=self.dtype,
                               generator=self.generator)
            flipped = torch.where(noise < logits.sigmoid(), 1 - self.state, self.state)
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

    # ------------------------------------------------------------------- run
    def run(self, deadline, decode, *, max_blocks=None, skip_unchanged=False):
        steps = blocks = 0
        decode_seconds = 0.0
        block_time = 0.0
        self.improvements.zero_()
        decoded_at = -1
        while True:
            if max_blocks is not None and blocks >= max_blocks:
                break
            if perf_counter() + block_time >= deadline:
                break
            started = perf_counter()
            self.graph_exec.replay()
            steps += self.chunk
            blocks += 1
            if skip_unchanged:
                moved = int(self.improvements.item())
                if moved == decoded_at:
                    block_time = perf_counter() - started
                    continue
                decoded_at = moved
            decode(self.best_states)
            decode_seconds += perf_counter() - started
            block_time = perf_counter() - started
        return steps, decode_seconds, blocks

    # ---------------------------------------------------------------- decode
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
