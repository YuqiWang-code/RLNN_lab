"""Check the frequency operator and exchange legality on small graphs."""
from itertools import combinations
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))
from mis_v2_exchange import ExchangeGraph, node_mask, nodes
from mis_v2_frequency import SEGF, exchange_band, exchange_state, frequency_topology
from models import GCN


def edges_from(pairs):
    directed = [(u, v) for a, b in pairs for u, v in ((a, b), (b, a))]
    return torch.tensor(directed, dtype=torch.long).reshape(-1, 2).T


def sparse(edges, num_nodes):
    weights = torch.full((edges.size(1),), 0.5)
    return torch.sparse_coo_tensor(edges, weights, (num_nodes, num_nodes)).to_sparse_csr()


def frequency_check():
    size = 7
    edges = edges_from(((0, 2), (0, 3), (1, 3), (1, 4), (2, 5)))
    adj = sparse(edges, size)
    x = torch.tensor([[1., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 1., 0.]]).T
    topo = frequency_topology(edges, adj, size)
    features, release, released = exchange_state(x, adj, topo)
    actual = exchange_band(features, x, release, released, adj)
    ordinary = 2 * adj.to_dense()
    for particle in range(x.size(1)):
        chosen = x[:, particle]
        freeable = release[:, particle]
        relation = ordinary * (chosen[:, None] * freeable[None, :] + freeable[:, None] * chosen[None, :])
        degree = relation.sum(1, keepdim=True)
        signal = features[:, particle]
        expected = torch.where(degree > 0, signal - relation @ signal / degree.clamp_min(1), 0)
        torch.testing.assert_close(actual[:, particle], expected)
    model = SEGF(16, 2, 1.02, "segf")
    logits = model(x, edges, adj, topo)
    for particle in range(x.size(1)):
        torch.testing.assert_close(logits[:, particle:particle + 1], model(x[:, particle:particle + 1], edges, adj))
    order = torch.tensor([6, 3, 5, 0, 2, 4, 1])
    permuted_edges = order.argsort()[edges]
    torch.testing.assert_close(logits[order], model(x[order], permuted_edges, sparse(permuted_edges, size)))
    batch_edges = torch.cat((edges, edges + size), dim=1)
    torch.testing.assert_close(torch.cat((logits, logits)), model(x.repeat(2, 1), batch_edges, sparse(batch_edges, 2 * size)))
    baseline = GCN(1, 16, 2, 1)
    model.load_state_dict(baseline.state_dict(), strict=False)
    with torch.no_grad():
        model.adapter.alpha_low.zero_()
        model.adapter.alpha_high.zero_()
        torch.testing.assert_close(model(x, edges, adj), baseline(x.unsqueeze(-1), edges).squeeze(-1))


def exchange_check():
    rng = random.Random(17)
    size = 10
    found = [0, 0]
    for _ in range(80):
        edges = edges_from(pair for pair in combinations(range(size), 2) if rng.random() < 0.35)
        graph = ExchangeGraph(size, edges)
        selected = graph.extend(0)
        outside = list(nodes(graph.all_nodes ^ selected))
        for remove in (1, 2):
            expected = set()
            for removed in combinations(nodes(selected), remove):
                retained = selected & ~node_mask(removed)
                for added in combinations(outside, remove + 1):
                    if graph.feasible(retained | node_mask(added)):
                        expected.add((removed, added))
            actions = graph.actions(selected, larger=remove == 2)
            assert {(a.removed, a.added) for a in actions} == expected
            for action in actions:
                result = graph.apply(selected, action)
                assert result.bit_count() > selected.bit_count() and graph.feasible(result)
                assert graph.features(selected, action) == graph.features(
                    selected, action._replace(added=action.added[::-1]))
                found[remove - 1] += 1
            if actions:
                break
    assert all(found), found


if __name__ == "__main__":
    torch.set_num_threads(1)
    frequency_check()
    exchange_check()
    print("SEGF operator, particle independence, block isolation, zero adapter and legal exchanges passed")
