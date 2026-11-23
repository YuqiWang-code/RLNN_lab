"""v9: density-compensated scale-space operator for independent-set search.

Why the frozen search plateaus
------------------------------
``models/models.py`` aggregates with ``D~^{-1/2} (A+I) D~^{-1/2}`` (``models/models.py:46``)
and stacks five of those layers (``NUM_L=5``, ``NUM_H=128``).  On the author RB graphs
the maximum out-degree is 104 (``capacity = 105`` in ``padded_ref0.json``, ``nodes=264``,
so the mean degree is ~80).  A normalised *mean* aggregation moves a node-dependent
quantity towards the neighbourhood average at rate ``d^{-(k-1)/2}``: after three hops
the contrast between two nodes is ~1/80 of the one-hop contrast, and after five it is
noise.  The five-layer network therefore behaves like a one-hop rule
``(x_i, selected-neighbour count) -> logit``, which has a single fixed point, so the
20 search particles all end in the same solution and every "diversify" patch
(temperature, kicks, refill, global token, exchange head) measures as zero.

What this operator does instead
-------------------------------
Read the hops *explicitly* as counts, and rescale each hop so that every scale enters
with unit spread instead of decaying or exploding:

    c_k(i) = sum_{j in N_k(i)} x_j                      (count, not mean)
    e_k    = (c_k - mean_i c_k) / std_i c_k             (per hop, over real rows only)
    h(i)   = sum_k w_k(tau) e_k(i),   w_k = e^-tau tau^k / k!   (heat kernel, tau learned)
    logit  = MLP([w_1 e_1, ..., w_K e_K, x, log deg])

Both extremes are real: on the author graphs the mean form loses 90% of its contrast
over three hops (``check_v9_scale.py`` measures 0.1001x) while the raw count form grows
3576x, so a five-hop read-out needs per-hop rescaling either way.  The state ``x`` is
passed through unchanged, so the operator never has to reconstruct it.

The one-hop count is recovered from the normalised matrix *without a second matrix*:
because ``adj = D~^{-1/2} (A+I) D~^{-1/2}``,

    (A x)_i = sqrt(d~_i) * (adj @ (sqrt(d~) * x))_i - x_i

The un-normaliser ``sqrt(d~)`` comes from the edge list (one ``bincount``) or, in the
padded engine, from the row degrees it already computes when it writes the weights.
Note that ``adj @ 1`` is *not* ``sqrt(d~)`` -- the normalisation mixes both endpoints --
so the operator never guesses it from the matrix.  The interface stays identical to
``GCN`` (``forward(x, edge_index, adj)``), so the eager and captured engines need no
change at all; the padded engine passes the vector it already has.
"""
import torch
from torch import nn

from models import edge2adj


class _Residual(nn.Module):
    """``x + relu(linear(relu(linear(x))))``: keeps the signal scale through depth.

    A plain five-layer MLP on a unit-variance descriptor starts with logits ~60x
    smaller than the residual ``GCN`` stack it replaces, because every ReLU layer
    halves the spread.  The skip keeps the descriptor scale, so the two operators
    start from comparable logits and the comparison is about the read-out only.
    """

    def __init__(self, width):
        super(_Residual, self).__init__()
        self.block = nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                   nn.Linear(width, width), nn.ReLU())

    def forward(self, x):
        return x + self.block(x)

    def reset_parameters(self):
        for module in self.block:
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class ScaleSpaceGCN(nn.Module):
    """Multi-hop count read-out with a learned heat-kernel scale, then a small MLP."""

    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.5,
                 hops=3, tau=2.0):
        super(ScaleSpaceGCN, self).__init__()
        if in_channels != 1:
            raise ValueError('ScaleSpaceGCN reads a scalar state per node')
        self.hops = hops
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        self.gain = nn.Parameter(torch.ones(hops))
        # Per-node descriptor: K rescaled hop counts, the state, and log degree.
        self.body = nn.Sequential(
            nn.Linear(hops + 2, hidden_channels),
            nn.ReLU(),
            *(_Residual(hidden_channels) for _ in range(max(num_layers - 2, 0))),
        )
        self.head = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.tau.data.fill_(2.0)
        self.gain.data.fill_(1.0)
        for module in self.body:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
            elif isinstance(module, _Residual):
                module.reset_parameters()
        self.head.reset_parameters()

    def heat(self):
        """Heat-kernel weights over hops 1..K, scaled by a learned per-hop gain."""
        tau = self.tau.clamp(0.05, 20.0)
        index = torch.arange(1, self.hops + 1, dtype=tau.dtype, device=tau.device)
        log_weight = -tau + index * tau.log() - torch.lgamma(index + 1.0)
        return torch.exp(log_weight) * self.gain

    def forward(self, x, edge_index=None, adj=None, scale=None):
        n = x.shape[0]
        state = x.reshape(n, -1)
        dtype = state.dtype
        if adj is None:
            adj = edge2adj(edge_index, n)
        if scale is None:
            if edge_index is None:
                raise ValueError('arch scale needs either edge_index or the sqrt(degree+1) vector')
            # ``edge2adj`` normalises with (degree + self loop) ** -0.5, so the
            # matching un-normaliser is sqrt(degree + 1).  Note that ``adj @ 1``
            # is *not* this vector: the normalisation mixes both endpoints.
            degree = torch.bincount(edge_index[0], minlength=n).to(dtype) + 1.0
            scale = degree.sqrt().unsqueeze(-1)
        # Row ``i`` is a real node iff it has at least one edge: every real node of
        # these instances has degree >= 1, so ``scale == 1`` marks the padding of
        # the fixed-shape buffer.  Statistics are taken over real rows only, which
        # keeps a padded graph identical to the unpadded one.
        valid = (scale > 1.0 + 1e-4).to(dtype)
        population = valid.sum(dim=0, keepdim=True).clamp_min(1.0)

        fields = []
        field = state
        for _ in range(self.hops):
            field = scale * torch.spmm(adj, scale * field) - field
            fields.append(field)

        weight = self.heat()
        descriptor = []
        for index, field in enumerate(fields):
            # Counts grow by roughly the walk count per hop (three orders of
            # magnitude over three hops on a mean-degree-80 graph), so a raw
            # multi-hop read-out is dominated by its deepest channel.  Standardise
            # each hop over real nodes: every scale then enters with unit spread
            # and a learnable relative weight, instead of the deepest hop winning
            # by magnitude alone.
            centre = (field * valid).sum(dim=0, keepdim=True) / population
            variance = ((field - centre) ** 2 * valid).sum(dim=0, keepdim=True) / population
            spread = variance.clamp_min(0.0).sqrt()
            standardised = (field - centre) / spread.where(spread > 1e-6, torch.ones_like(spread))
            descriptor.append(weight[index] * standardised * valid)

        descriptor.append(state)
        descriptor.append((scale.log() * valid).expand(-1, state.shape[1]))
        flat = torch.stack(descriptor, dim=-1).reshape(n * state.shape[1], -1)
        return self.head(self.body(flat)).reshape(state.shape).reshape(x.shape)
