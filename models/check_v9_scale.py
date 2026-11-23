"""Sanity checks for the v9 scale-space operator (``models/mis_v9_scale.py``).

The v9 argument rests on four claims about the operator; each is measured here on
a synthetic graph shaped like an author instance (n=264, mean degree ~80, which is
what ``capacity = 105`` at ``nodes = 264`` implies):

1. the un-normaliser is right: the sqrt(degree + 1) vector applied through the
   normalised matrix reproduces ``A @ x`` exactly;
2. the mean read-out decays with the hop index (this *is* the D1 defect that
   motivated v9), while the raw count read-out explodes -- so neither is usable
   without rescaling, which is why each hop is standardised;
3. the model runs on both call paths the engines use, with finite output;
4. padding is inert: a 200-node graph embedded in a 264-row buffer produces the
   same logits on its real rows as the unpadded call.

    python models/check_v9_scale.py
"""
import torch

from models import edge2adj
from mis_v9_scale import ScaleSpaceGCN


def random_dense_graph(nodes, degree, generator):
    """Symmetric simple graph with approximately ``degree`` neighbours per node."""
    adjacency = torch.zeros(nodes, nodes)
    for source in range(nodes):
        targets = torch.randperm(nodes, generator=generator)[:degree]
        targets = targets[targets != source]
        adjacency[source, targets] = 1.0
    adjacency = ((adjacency + adjacency.t()) > 0).float()
    adjacency.fill_diagonal_(0.0)
    return adjacency


def hop_fields(adjacency, edge_index, state, hops):
    """The two read-outs the operator has to choose between, hop by hop."""
    nodes = adjacency.shape[0]
    scale = (torch.bincount(edge_index[0], minlength=nodes).float() + 1.0).sqrt().unsqueeze(-1)
    adj = edge2adj(edge_index, nodes)
    mean_form, count_form = state, state
    means, counts = [], []
    for _ in range(hops):
        mean_form = torch.spmm(adj, mean_form)
        count_form = scale * torch.spmm(adj, scale * count_form) - count_form
        means.append(float(mean_form.std()))
        counts.append(float(count_form.std()))
    return means, counts


def main():
    generator = torch.Generator().manual_seed(0)
    nodes, degree = 264, 80
    adjacency = random_dense_graph(nodes, degree, generator)
    edge_index = adjacency.nonzero().t().contiguous()
    adj = edge2adj(edge_index, nodes)

    model = ScaleSpaceGCN(1, 64, 5, 1, hops=3)
    model.reset_parameters()
    model.eval()

    state = (torch.rand(nodes, 1, generator=generator) < 0.08).float()
    with torch.no_grad():
        scale = (torch.bincount(edge_index[0], minlength=nodes).float() + 1.0).sqrt()
        mine = scale.unsqueeze(-1) * torch.spmm(adj, scale.unsqueeze(-1) * state) - state
        print(f"1) one-hop count vs A@x: max |diff| = {(mine - adjacency @ state).abs().max():.3e}, "
              f"max count {float(mine.max()):.1f}")

        means, counts = hop_fields(adjacency, edge_index, state, 3)
        print("2) hop   mean-form std   count-form std")
        for hop, (mean_std, count_std) in enumerate(zip(means, counts), start=1):
            print(f"   {hop}     {mean_std:.6f}        {count_std:.4f}")
        print(f"   mean form decays {means[-1] / means[0]:.4f}x over three hops (D1), "
              f"count form grows {counts[-1] / counts[0]:.0f}x -> both rescaled by the operator")

        for particles in (1, 20):
            batch = (torch.rand(nodes, particles, generator=generator) < 0.08).float()
            logits = model(batch.unsqueeze(-1), edge_index, adj)
            explicit = model(batch.unsqueeze(-1), None, adj, scale.unsqueeze(-1))
            print(f"3) particles={particles}: shape {tuple(logits.shape)} "
                  f"finite {bool(torch.isfinite(logits).all())} "
                  f"std {float(logits.std()):.4f} "
                  f"| edge_index path == explicit scale path: "
                  f"{bool(torch.allclose(logits, explicit, atol=1e-6))}")

        # Padding must not move the real rows: the operator masks on degree.
        smaller = 200
        sub = adjacency[:smaller, :smaller]
        keep = (sub.sum(dim=1) > 0) & (sub.sum(dim=0) > 0)
        index = keep.nonzero().reshape(-1)[:smaller]
        sub = adjacency[index][:, index]
        sub_edges = sub.nonzero().t().contiguous()
        sub_state = (torch.rand(smaller, 20, generator=generator) < 0.08).float()
        small_logits = model(sub_state.unsqueeze(-1), sub_edges, edge2adj(sub_edges, smaller))

        wide_state = torch.zeros(nodes, 20)
        wide_state[:smaller] = sub_state
        wide_adjacency = torch.zeros(nodes, nodes)
        wide_adjacency[:smaller, :smaller] = sub
        wide_edges = wide_adjacency.nonzero().t().contiguous()
        wide_scale = torch.ones(nodes, 1)
        wide_scale[:smaller] = (torch.bincount(wide_edges[0], minlength=nodes).float() + 1.0).sqrt()[:smaller].unsqueeze(-1)
        wide_logits = model(wide_state.unsqueeze(-1), None, edge2adj(wide_edges, nodes), wide_scale)
        print(f"4) padding: 200 real nodes in a {nodes}-row buffer, "
              f"max |real-row diff| = {(wide_logits[:smaller] - small_logits).abs().max():.3e}")
    print("ok")


if __name__ == "__main__":
    main()
