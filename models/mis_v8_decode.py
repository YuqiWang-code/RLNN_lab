"""The A.3 decode as a replayable CUDA-graph chunk.

Once the search was captured (``mis_v8_capture.py``) the decode became the
bottleneck: ``outputs/v8_macro/positioning/capture_budget/capture_budget_ref0.csv``
puts it at 71.5 % of search time at 0.5 s and 86.6 % at 1.0 s.  ``probe_decode_cost.py``
shows why: ``decode_gradient_flip`` is a host-driven loop that ends every
iteration in a device-to-host read (``if not (gain > 0).any(): break``) and pays
0.44-1.10 ms for GPU work that a captured chunk replays in 0.055 ms.

This module keeps the loop but moves the work into captured chunks of ``chunk``
iterations.  Over-running is safe and exact: once a node's gain stops improving
its update is ``where(gain > 0, 1 - x, x)``, i.e. the state does not move, so
further iterations are no-ops.  A chunk therefore reports whether *any* particle
flipped, and the driver replays chunks until one reports none.  The result is
bit-identical to the eager loop (asserted in ``check_decode_capture.py``).

What is left is one host read per chunk instead of per iteration, and the
arithmetic runs without launch overhead.

    decoder = CapturedDecoder(problem, adj, bias, template, chunk=8)
    solutions = decoder.decode(best_states)   # (nodes, particles), same ints as eager
"""
import numpy as np
import torch

from eval_v2 import sync


class CapturedDecoder:
    """A.3 local search for one (graph, particle-count) shape, captured once."""

    def __init__(self, problem, adjacency, bias, template, *, chunk=8):
        if not torch.cuda.is_available():
            raise RuntimeError("captured decode requires CUDA")
        self.problem = problem
        self.chunk = chunk
        self.device = template.device
        self.dtype = template.dtype
        self.nodes, self.particles_count = template.shape
        self.solutions = template.clone()
        self.gradient = torch.empty_like(self.solutions)
        self.delta = torch.empty_like(self.solutions)
        self.gain = torch.empty(self.particles_count, device=self.device, dtype=self.dtype)
        self.node = torch.empty(self.particles_count, device=self.device, dtype=torch.long)
        self.selected = torch.empty(1, self.particles_count, device=self.device, dtype=self.dtype)
        self.updates = torch.empty(1, self.particles_count, device=self.device, dtype=self.dtype)
        self.flag = torch.zeros(1, device=self.device, dtype=torch.bool)
        self.particles = torch.arange(self.particles_count, device=self.device)
        self.twobeta = 2 * problem.beta
        self.bias = bias
        self.adjacency = adjacency
        # The warm-up and the recording both advance ``solutions``; put the caller's
        # matrix back so the captured object starts from where it was handed over.
        snapshot = self.solutions.clone()
        self._build()
        self.solutions.copy_(snapshot)
        self.max_replays = 8 * self.nodes // chunk + 8

    def _iteration(self):
        """One A.3 iteration, no host read: the block body of the captured graph."""
        torch.add(torch.mul(self.adjacency @ self.solutions, self.twobeta), self.bias,
                  out=self.gradient)
        torch.mul(self.gradient, (2 * self.solutions - 1) / 2, out=self.delta)
        torch.max(self.delta, dim=0, out=(self.gain, self.node))
        torch.gather(self.solutions, 0, self.node.unsqueeze(0), out=self.selected)
        torch.where(self.gain > 0, 1 - self.selected, self.selected, out=self.updates)
        self.solutions.index_put_((self.node, self.particles), self.updates.squeeze(0))

    def _body(self):
        for _ in range(self.chunk):
            self._iteration()
            torch.logical_or(self.flag, (self.gain > 0).any(dim=0, keepdim=True), out=self.flag)

    def _build(self):
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._body()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.executor = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.executor, stream=stream):
            self._body()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        sync(self.device)

    @torch.no_grad()
    def decode(self, states, *, max_replays=None):
        """Decode ``states``; returns the (nodes, particles) int solution matrix."""
        self.solutions.copy_(states)
        limit = self.max_replays if max_replays is None else max_replays
        replays = 0
        while True:
            self.flag.zero_()
            self.executor.replay()
            replays += 1
            if not bool(self.flag.item()):
                break
            if replays >= limit:
                raise RuntimeError(f"decode did not converge in {limit} replays")
        return self.solutions

    @torch.no_grad()
    def decode_best(self, states):
        """Decode ``states`` and return only the largest particle as a bitset.

        ``solution_masks`` ships all 20 columns to the host and packs them there.
        The search only ever uses the largest one, so the argmax happens on the
        device and a single column crosses the bus -- about a third of the cost of
        the full packing, and it keeps the decode callable as ``masks_of``.
        """
        solutions = self.decode(states)
        sizes = solutions.sum(dim=0)
        _, index = torch.max(sizes, dim=0)
        column = solutions[:, int(index)].to(torch.uint8)
        packed = np.packbits(column.cpu().numpy(), bitorder="little")
        return int.from_bytes(packed.tobytes(), "little")
