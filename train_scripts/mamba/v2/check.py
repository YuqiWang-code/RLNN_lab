"""CPU checks for zero-dose identity, direction controls and the v2 feature path."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

from mis_static_prior import bias_stats, oracle_bias, random_bias
from mis_v1 import adjacency
from mis_v7_context import V7StaticContextGCN
from models import GCN
from utils import build_batch
from v7_structure import structure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mamba", action="store_true", help="Also exercise the CUDA Mamba encoder")
    args = parser.parse_args()
    torch.manual_seed(1017)
    edges = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    graph = SimpleNamespace(num_nodes=6, idx=0, edge_index=edges, edge_weight=torch.full((6,), .5))
    graph.x = torch.ones(6, 1)
    graph.b = -graph.x
    batch = build_batch([graph, graph])
    assert len(batch.graph_ptr) - 1 == 2 and batch.batch.unique().tolist() == [0, 1]
    adj = adjacency(graph)
    x = torch.randint(2, (6, 4)).float()
    base = GCN(1, 8, 2, 1).eval()
    model = V7StaticContextGCN(8, 2, "static_centered", 16, 16).eval()
    missing, extra = model.load_state_dict(base.state_dict(), strict=False)
    assert not extra and all(key.startswith("adapter.") for key in missing)
    assert model.delta_limit == (.25, .125)
    assert V7StaticContextGCN(8, 2, "static_centered", 16, 16, .5).delta_limit == (.25, .5)
    assert V7StaticContextGCN(8, 2, "static_centered", 16, 16, .5,
                              static_node_limit=1.).delta_limit == (.25, 1.)
    adapter = model.adapter
    with torch.no_grad():
        adapter.constants.fill_(.5)
        adapter.alpha.fill_(.5)
        topology = adapter.topology(graph, adj)
        constant = adapter.action_bias(topology, (.25, 0.))
        active = adapter.action_bias(topology, (.25, .125))
        assert torch.allclose((active-constant).mean(0), torch.zeros(2), atol=1e-7)
        zero = topology._replace(cached_action_bias=adapter.action_bias(topology, (0., 0.)))
        assert torch.equal(base(x.unsqueeze(-1), edges, topology.normalized).squeeze(-1), model(x, edges, adj, zero))
        random = random_bias(6, 7311, .0107, x.device)
        assert torch.equal(random, random_bias(6, 7311, .0107, x.device))
        assert torch.allclose(random.mean(0), torch.zeros(2), atol=1e-7)
        assert abs(bias_stats(random)["bias_rms"] - .0107) < 1e-7
        target = oracle_bias(6, [0, 2, 4], 1., x.device)
        expected = torch.tensor([[1., -1.], [-1., 1.]]).repeat(3, 1)
        assert torch.equal(target, expected)
        oracle_model = V7StaticContextGCN(8, 2, "static_centered", 16, 16,
                                          static_node_limit=2., peak_normalize=True)
        oracle_model.adapter.alpha.fill_(torch.atanh(torch.tensor(.5)))
        batch_adj = adjacency(batch)
        topology = oracle_model.adapter.topology(batch, batch_adj)
        bias = oracle_model.adapter.action_bias(topology)
        for part in bias.split(6):
            torch.testing.assert_close(part.abs().amax(), torch.tensor(1.))
            torch.testing.assert_close(part.mean(0), torch.zeros(2), atol=1e-6, rtol=0)

    features, orders = structure(edges.numpy(), graph.num_nodes, 1017, order="bfs", views=2)
    sparse, sparse_orders = structure(edges.numpy(), graph.num_nodes, 1017, order="bfs", views=2, sparse=True)
    np.testing.assert_array_equal(features, sparse)
    for left, right in zip(orders, sparse_orders, strict=True):
        np.testing.assert_array_equal(left, right)
    for encoder in ("mlp", "matched-mlp", "msg"):
        model = V7StaticContextGCN(8, 2, "static_centered", 16, 16,
                                   feature_dim=8, static_encoder=encoder)
        topology = model.adapter.topology(graph, adj)
        if model.adapter.encoder is not None:
            assert model.adapter.encoder.normalize(topology.features) is topology.features
        model(x, edges, adj, topology).sum().backward()
        assert model.adapter.alpha.grad is not None
    print("Static v2 CPU checks passed")
    if args.mamba:
        graph.edge_index = graph.edge_index.cuda()
        graph.edge_weight = graph.edge_weight.cuda()
        adj = adjacency(graph)
        model = V7StaticContextGCN(8, 2, "static_centered", 16, 16,
                                   feature_dim=8, static_encoder="mamba").cuda()
        model.load_state_dict(model.state_dict())
        topology = model.adapter.topology(graph, adj)
        model.adapter.alpha.data.fill_(.5)
        logits = model(x.cuda(), graph.edge_index, adj, topology)
        logits.square().mean().backward()
        grad = model.adapter.encoder.sequence.in_proj.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)
        print("Static v2 CUDA Mamba forward/backward passed")


if __name__ == "__main__":
    main()
