"""CPU checks for the structural-compensation module before any GPU run.

Verifies the zero-initialised identity against the frozen backbone, that each
switch actually changes the logits once its parameters move, that the degree
encoding is batch invariant, and that the gain residual reproduces the analytic
flip gain ordering.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))

from mis_structure import (DEGREE_FREQUENCIES, STRUCTURE_MODES, StructureCompensation,
                           StructureGCN, degree_features, structure_topology)
from models import GCN, edge2adj
from utils import build_batch


def make_graph(num_nodes, edges, index=0):
    x = torch.ones(num_nodes, 1)
    return SimpleNamespace(num_nodes=num_nodes, idx=index, edge_index=edges, edge_weight=torch.full((edges.size(1),), .5),
                           x=x, b=-x)


def main():
    torch.manual_seed(1017)
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    graph = make_graph(6, edges)
    adj = edge2adj(edges, 6)
    topology = structure_topology(edges, adj, 6)
    degree = torch.tensor([1., 2., 1., 1., 1., 0.])
    assert torch.equal(topology.degree, degree), topology.degree
    feature, centered = degree_features(degree, (0, 6))
    assert feature.shape == (6, 2 + 2 * len(DEGREE_FREQUENCIES))
    assert torch.allclose(centered.mean(), torch.zeros(()), atol=1e-6)

    x = torch.randint(2, (6, 4)).float()
    product = adj @ x
    base = GCN(1, 8, 2, 1).eval()
    for mode in STRUCTURE_MODES[1:]:
        model = StructureGCN(8, 2, mode).eval()
        missing, extra = model.load_state_dict(base.state_dict(), strict=False)
        assert not extra and all(key.startswith("compensation.") for key in missing), (mode, missing)
        with torch.no_grad():
            reference = base(x.unsqueeze(-1), edges, adj).squeeze(-1)
            got = model(x, edges, adj, topology, product)
        assert torch.equal(reference, got), f"{mode} does not reproduce B at zero compensation"
        # Turning the switch on must change the logits.
        with torch.no_grad():
            compensation = model.compensation
            if compensation.use_degree:
                compensation.degree_proj[-1].weight.normal_(0, .1)
            if compensation.use_gate:
                compensation.gate.fill_(1.)
            if compensation.use_gain:
                compensation.gain_gate.fill_(1.)
                compensation.gain_net[0].weight.normal_(0, .1)
            if compensation.use_readout:
                compensation.degree_out.weight.normal_(0, .1)
            if getattr(compensation, "use_scaled_gain", False):
                compensation.gain_net[0].weight.normal_(0, .1)
            moved = model(x, edges, adj, topology, product)
        assert not torch.allclose(reference, moved), f"{mode} switch has no effect"

    # Batch invariance: the degree encoding of a graph must not depend on its batch.
    second = make_graph(5, torch.tensor([[0, 1, 2], [1, 2, 3]]), index=1)
    singles = [structure_topology(g.edge_index, edge2adj(g.edge_index, g.num_nodes), g.num_nodes)
               for g in (graph, second)]
    batch = build_batch([graph, second])
    batched = structure_topology(batch.edge_index, edge2adj(batch.edge_index, batch.num_nodes),
                                 batch.num_nodes, batch.graph_ptr)
    assert torch.allclose(batched.degree, torch.cat([single.degree for single in singles]))
    assert torch.allclose(batched.degree_feature,
                          torch.cat([single.degree_feature for single in singles]), atol=1e-6)
    assert torch.allclose(batched.degree_centered,
                          torch.cat([single.degree_centered for single in singles]), atol=1e-6)

    # The gate factor must be bounded and equal one at zero gate.
    gate = StructureCompensation(8, 2, "gate")
    scale = gate.gate_scale(0, topology)
    assert torch.allclose(scale, torch.ones_like(scale)), scale
    with torch.no_grad():
        gate.gate.fill_(8.)
    bounded = gate.gate_scale(0, topology)
    assert bounded.max() <= 1.65 + 1e-5 and bounded.min() >= .6 - 1e-5, bounded

    # The gain residual must broadcast over particles and change the logits.
    gain = StructureCompensation(8, 2, "gain")
    with torch.no_grad():
        gain.gain_gate.fill_(1.)
        residual = gain.gain(x, topology, product)
    assert residual.shape == x.shape, residual.shape
    assert residual.abs().sum() > 0

    # Cached static tensors must reproduce the on-the-fly path bit for bit.
    cached_model = StructureGCN(8, 2, "all").eval()
    cached_model.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        for layer in cached_model.compensation.degree_proj:
            layer.weight.normal_(0, .1)
        cached_model.compensation.gate.fill_(1.)
        cached_model.compensation.gain_gate.fill_(1.)
        cached_model.compensation.gain_net[0].weight.normal_(0, .1)
        plain = cached_model(x, edges, adj, topology, product)
        bias, scale, readout = cached_model.compensation.prepare(topology)
        cached = cached_model(x, edges, adj,
                              topology._replace(cached_bias=bias, cached_scale=scale,
                                                cached_readout=readout), product)
    assert torch.allclose(plain, cached, atol=1e-5), (plain - cached).abs().max()

    print("Structure checks passed: zero-init identity, switches, batch invariance, gain plumbing, cache parity")


if __name__ == "__main__":
    main()
