"""Check v4 filter semantics, backbone compatibility and independent graph batching."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))
from models import GCN
from mis_v4_frequency import V4FrequencyGCN, V4_MODES, lazy_walk, state_bands, v4_topology


def graph(nodes, pairs, device):
    edges = torch.tensor(pairs + [(j, i) for i, j in pairs],
                         dtype=torch.long, device=device).t().contiguous()
    adj = torch.sparse_coo_tensor(edges, torch.full((edges.shape[1],), 0.5, device=device),
                                  (nodes, nodes)).to_sparse_csr()
    return edges, adj, v4_topology(edges, adj, nodes)


def close(actual, expected, atol=1e-6, rtol=1e-5):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def run(device):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    nodes = 7
    pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    edges, adj, topology = graph(nodes, pairs, device)
    x = torch.tensor([[1, 0, 1], [0, 1, 1], [0, 0, 0], [1, 0, 0],
                      [0, 1, 0], [1, 0, 1], [0, 1, 1]], device=device).float()
    a = topology.binary.to_dense()
    d = a.sum(1, keepdim=True)
    transition = a / d.clamp_min(1)
    transition[d.squeeze(1) == 0] = torch.eye(nodes, device=device)[d.squeeze(1) == 0]
    p = 0.5 * (torch.eye(nodes, device=device) + transition)
    details = state_bands(x, topology)
    dense_z = torch.stack((x, (a @ x).log1p()), -1)
    dense_pz = (p @ dense_z.flatten(1)).reshape_as(dense_z)
    dense_ppz = (p @ dense_pz.flatten(1)).reshape_as(dense_z)
    close(details["c"], 2 * (adj @ x), atol=0, rtol=0)
    close(details["r"], x * (a @ ((x == 0) & ((a @ x) == 1)).float()), atol=0, rtol=0)
    close(details["z"], dense_z)
    close(details["pz"], dense_pz)
    close(details["ppz"], dense_ppz)
    close(details["L2"] + details["B0"] + details["B1"], details["z"])
    for key in ("B0", "B1"):
        close(details[key][5:], torch.zeros_like(details[key][5:]), atol=0, rtol=0)
    constant = torch.tensor([2.0, -3.0], device=device).expand(nodes, 3, 2)
    close(lazy_walk(constant, topology), constant, atol=0, rtol=0)
    close(lazy_walk(lazy_walk(constant, topology), topology), constant, atol=0, rtol=0)
    all_selected = state_bands(torch.ones_like(x), topology)
    assert all_selected["B0"][..., 1].abs().max() > 0
    assert not torch.equal(state_bands(1 - x, topology)["c"], details["c"])

    base = GCN(1, 128, 5, 1).to(device).eval()
    model = V4FrequencyGCN(128, 5).to(device).eval()
    missing, extra = model.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(key.startswith("adapter.") for key in missing) and not extra
    with torch.no_grad():
        expected = base(x.unsqueeze(-1), edges, topology.normalized).squeeze(-1)
        close(model(x, edges, adj, topology, disable_adapter=True), expected, atol=0, rtol=0)
        alpha = (model.adapter.alpha_low.item(), model.adapter.alpha_high.item())
        assert all(abs(value - 0.01) < 1e-8 for value in alpha)
        model.adapter.alpha_low.zero_()
        model.adapter.alpha_high.zero_()
        close(model(x, edges, adj, topology), expected, atol=0, rtol=0)
        model.adapter.alpha_low.fill_(alpha[0])
        model.adapter.alpha_high.fill_(alpha[1])

    counts = {}
    for mode in V4_MODES:
        arm = V4FrequencyGCN(128, 5, mode=mode).to(device)
        counts[mode] = sum(parameter.numel() for parameter in arm.parameters())
        assert torch.isfinite(arm(x, edges, adj, topology)).all()
    assert set(counts.values()) == {170513}
    assert sum(parameter.numel() for parameter in model.adapter.parameters()) == 5008
    message = V4FrequencyGCN(128, 5, mode="v4a_message").to(device)
    info = message.adapter.inspect(x, adj, topology)
    raw_z, raw_pz = info["high_input"].chunk(2, -1)
    close(raw_z - raw_pz, details["B0"])
    close(raw_pz - info["low_input"], details["B1"])
    model.adapter.gate_override = 0.5
    model.adapter.low_override = False
    info = model.adapter.inspect(x, adj, topology)
    close(info["gates"], torch.full_like(info["gates"], 0.5), atol=0, rtol=0)
    close(info["delta_low"], torch.zeros_like(info["delta_low"]), atol=0, rtol=0)
    model.adapter.gate_override = model.adapter.low_override = None

    order = torch.tensor([4, 6, 1, 0, 5, 2, 3], device=device)
    inverse = torch.argsort(order)
    permuted_edges = inverse[edges]
    permuted_adj = torch.sparse_coo_tensor(
        permuted_edges, torch.full((edges.shape[1],), 0.5, device=device), adj.shape,
    ).to_sparse_csr()
    permuted_topology = v4_topology(permuted_edges, permuted_adj, nodes)
    output = model(x, edges, adj, topology)
    close(model(x[order], permuted_edges, permuted_adj, permuted_topology), output[order], atol=1e-5)
    particle_order = torch.tensor([2, 0, 1], device=device)
    close(model(x[:, particle_order], edges, adj, topology), output[:, particle_order], atol=1e-5)
    for particles in (1, 20, 64, 1000):
        repeated = x[:, :1].expand(-1, particles).contiguous()
        scored = model(repeated, edges, adj, topology)
        assert scored.shape == (nodes, particles)
        close(scored, output[:, :1].expand_as(scored), atol=1e-5)

    other_edges, other_adj, other_topology = graph(4, [(0, 1), (1, 2)], device)
    other_x = 1 - x[:4]
    batched_edges = torch.cat((edges, other_edges + nodes), 1)
    batched_adj = torch.sparse_coo_tensor(
        batched_edges, torch.full((batched_edges.shape[1],), 0.5, device=device), (11, 11),
    ).to_sparse_csr()
    batched_topology = v4_topology(batched_edges, batched_adj, 11)
    batched_model, separate_model = copy.deepcopy(model), copy.deepcopy(model)
    batched = batched_model(torch.cat((x, other_x)), batched_edges, batched_adj, batched_topology)
    separate = torch.cat((separate_model(x, edges, adj, topology),
                          separate_model(other_x, other_edges, other_adj, other_topology)))
    close(batched, separate, atol=1e-5)
    batched.square().mean().backward()
    separate.square().mean().backward()
    gradient_error = 0.0
    for (name, parameter), (_, reference) in zip(batched_model.named_parameters(), separate_model.named_parameters()):
        close(parameter.grad, reference.grad, atol=1e-5)
        gradient_error = max(gradient_error, (parameter.grad - reference.grad).abs().max().item())
        if name.startswith("adapter."):
            assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name

    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.float16), torch.no_grad():
            info = model.adapter.inspect(x.half(), adj.half(), topology)
            assert info["z"].dtype == torch.float32
            assert info["c"].dtype == torch.float32
            assert torch.isfinite(model(x.half(), edges, adj.half(), topology)).all()
            no_adapter = model(x.half(), edges, adj.half(), topology, disable_adapter=True)
            original = base(x.half().unsqueeze(-1), edges, topology.normalized).squeeze(-1)
            close(no_adapter, original, atol=0, rtol=0)
    return {"status": "passed", "device": str(device), "params": counts,
            "adapter_params": 5008, "batch_gradient_max_abs_difference": gradient_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(run(torch.device(args.device)), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)
