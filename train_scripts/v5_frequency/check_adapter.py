"""Check scalar filters, backbone compatibility, graph batching and AMP."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))
from models import GCN
from mis_v4_frequency import lazy_walk
from mis_v5_frequency import V5DAFRGCN, V5_MODES, local_energy, v5_topology


def graph(nodes, pairs, device):
    edges = torch.tensor(pairs + [(j, i) for i, j in pairs], device=device, dtype=torch.long).reshape(-1, 2).T.contiguous()
    adj = torch.sparse_coo_tensor(edges, torch.full((edges.shape[1],), .5, device=device), (nodes, nodes)).to_sparse_csr()
    return edges, adj, v5_topology(edges, adj, nodes)


def run(device):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    close = torch.testing.assert_close
    pairs = [(0, 1), (0, 2), (1, 2), (3, 4)]
    edges, adj, topo = graph(7, pairs, device)
    x = torch.randint(2, (7, 3), device=device).float()
    dense = topo.binary.to_dense()
    close(topo.binary @ x, 2 * (adj @ x), atol=0, rtol=0)
    delta = local_energy(x, dense @ x, 1.02)

    def energy(state):
        return (state * (1.02 * (dense @ state) / 2 - 1)).sum(0)

    for i in range(7):
        y = x.clone()
        y[i] = 1 - y[i]
        close(energy(y) - energy(x), delta[i])
    u = torch.randn(7, 3, 2, device=device)
    low = lazy_walk(u, topo)
    high = u - low
    close(low + high, u)
    close(high[5:], torch.zeros_like(high[5:]), atol=0, rtol=0)
    for component in (slice(0, 3), slice(3, 5)):
        close((topo.degree[component, :, None] * high[component]).sum(0),
              torch.zeros_like(high[0]), atol=2e-6, rtol=0)
    constant = torch.full_like(u, 3)
    close(lazy_walk(constant, topo), constant, atol=0, rtol=0)
    close(lazy_walk(u, graph(7, [], device)[2]), u, atol=0, rtol=0)
    base = GCN(1, 128, 5, 1).to(device)
    model = V5DAFRGCN(128, 5).to(device)
    missing, extra = model.load_state_dict(base.state_dict(), strict=False)
    assert set(missing) == {"adapter." + name for name in model.adapter.state_dict()} and not extra
    reference = base(x.unsqueeze(-1), edges, topo.normalized).squeeze(-1)
    close(model(x, edges, adj, topo), reference, atol=1e-6, rtol=1e-6)
    for step in range(2):
        model.zero_grad()
        correction = model.adapter(x, topo)
        (correction * torch.arange(7, device=device)[:, None]).sum().backward()
        grads = [p.grad.abs().sum().item() for p in model.adapter.parameters()]
        if step == 0:
            assert all(g > 0 for g in grads[:2]) and all(g == 0 for g in grads[2:])
            with torch.no_grad():
                for p in (model.adapter.theta_low, model.adapter.theta_high):
                    p.add_(p.grad, alpha=-.1)
        else:
            assert all(g > 0 for g in grads[2:])
    params = {}
    for mode in V5_MODES:
        arm = V5DAFRGCN(128, 5, mode=mode).to(device)
        params[mode] = sum(p.numel() for p in arm.parameters())
        assert torch.isfinite(arm(x, edges, adj, topo)).all()
    assert set(params.values()) == {165617}
    order = torch.tensor([3, 6, 0, 2, 4, 5, 1], device=device)
    inverse = order.argsort()
    moved = graph(7, [(int(inverse[i]), int(inverse[j])) for i, j in pairs], device)
    output = model(x, edges, adj, topo)
    close(model(x[order], *moved), output[order], atol=1e-5, rtol=1e-5)
    close(model(x[:, [2, 0, 1]], edges, adj, topo), output[:, [2, 0, 1]], atol=1e-5, rtol=1e-5)
    for k in (1, 20, 64, 1000):
        with torch.no_grad():
            close(model(x[:, :1].repeat(1, k), edges, adj, topo), output[:, :1].expand(-1, k), atol=1e-5, rtol=1e-5)
    second = [(0, 1), (1, 2)]
    g2 = graph(4, second, device)
    gb = graph(11, pairs + [(i + 7, j + 7) for i, j in second], device)
    x2 = 1 - x[:4]
    batched, separate = copy.deepcopy(model), copy.deepcopy(model)
    yb = batched(torch.cat((x, x2)), *gb)
    ys = torch.cat((separate(x, edges, adj, topo), separate(x2, *g2)))
    close(yb, ys, atol=1e-5, rtol=1e-5)
    for m, y in ((batched, yb), (separate, ys)):
        m.zero_grad()
        y.square().mean().backward()
    for p, q in zip(batched.parameters(), separate.parameters()):
        close(p.grad, q.grad, atol=1e-5, rtol=1e-5)
    if device.type == "cuda":
        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.float16):
            y = model(x.half(), edges, adj.half(), topo)
            _, d = model.adapter(x.half(), topo, return_details=True)
        assert y.dtype == torch.float16 and d["counts"].dtype == torch.float32 and torch.isfinite(y).all()
        y.float().square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return dict(status="passed", device=str(device), params=params, adapter_params=112,
                learned_projection_gradients_after_first_update=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.dumps(run(torch.device(args.device)), indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result + "\n", encoding="utf-8")
    print(result)
