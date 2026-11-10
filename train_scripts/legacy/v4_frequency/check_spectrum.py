"""Check spectral centering and non-mutating v4 stage/intervention diagnostics."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))
from archived_frequency.frequency_diag_utils import (
    BANDS, intervention_details, override_logits, remove_stationary, spectral_basis,
    spectral_energy_extended, spectral_energy_many, trace_model, trace_sequential,
)


def close(actual, expected, atol=1e-10, rtol=1e-9):
    torch.testing.assert_close(torch.as_tensor(actual, dtype=torch.float64),
                               torch.as_tensor(expected, dtype=torch.float64), atol=atol, rtol=rtol)


def edges_from(pairs):
    return torch.tensor(pairs + [(j, i) for i, j in pairs], dtype=torch.long).t().contiguous()


def spectral_checks():
    edges = edges_from([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5)])
    basis = spectral_basis(edges, 7)
    assert int((basis[0] <= 1e-7).sum()) == 3
    signal = torch.randn(7, 4, 3, dtype=torch.float64)
    stats = spectral_energy_extended(signal, basis)
    close(sum(stats[f"raw_{name}_energy"] for _, _, name in BANDS), stats["total_energy"])
    close(sum(stats[f"raw_{name}_fraction"] for _, _, name in BANDS), 1.0)
    close(sum(stats[f"centered_{name}_fraction"] for _, _, name in BANDS), 1.0)
    close(stats["stationary_energy"] + stats["nonstationary_energy"], stats["total_energy"])
    centered = remove_stationary(signal, basis)
    centered_stats = spectral_energy_extended(centered, basis)
    close(centered_stats["total_energy"], stats["nonstationary_energy"])
    assert centered_stats["stationary_fraction"] < 1e-20
    for _, _, name in BANDS:
        close(centered_stats[f"raw_{name}_fraction"], stats[f"centered_{name}_fraction"])

    component_constants = torch.tensor([1, 1, 1, 2, 2, 2, 3], dtype=torch.float64)[:, None, None].expand(-1, 4, 3)
    const = spectral_energy_extended(component_constants, basis)
    close(const["stationary_fraction"], 1.0)
    assert const["weighted_global_constant_fraction"] < 1
    assert not const["valid_centered"] and const["centered_higher_fraction"] is None
    global_constant = spectral_energy_extended(torch.ones_like(component_constants), basis)
    close(global_constant["node_constant_fraction"], 1.0)
    close(global_constant["weighted_global_constant_fraction"], 1.0)
    assert not global_constant["valid_centered"]
    zero = spectral_energy_extended(torch.zeros_like(signal), basis)
    assert not zero["valid_total"] and not zero["valid_centered"]
    assert zero["raw_low_fraction"] is None

    permutation = torch.tensor([5, 0, 6, 3, 2, 1, 4])
    permuted = spectral_basis(torch.argsort(permutation)[edges], 7)
    permuted_stats = spectral_energy_extended(signal[permutation], permuted)
    for key in stats:
        if isinstance(stats[key], float) and stats[key] is not None:
            close(stats[key], permuted_stats[key])
    first = spectral_energy_extended(signal[:3], spectral_basis(edges_from([(0, 1), (1, 2), (0, 2)]), 3))
    second = spectral_energy_extended(signal[3:], spectral_basis(edges_from([(0, 1), (1, 2)]), 4))
    for key in ["total_energy", "stationary_energy", "nonstationary_energy"] + [f"raw_{name}_energy" for _, _, name in BANDS]:
        close(stats[key], first[key] + second[key])
    packed = spectral_energy_many({"a": signal, "b": signal * 2}, basis)
    close(packed["a"]["total_energy"], stats["total_energy"])
    close(packed["b"]["total_energy"], 4 * stats["total_energy"])

    block = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 128), nn.LayerNorm(128))
    inputs = torch.randn(7, 4, 4)
    parameters = {name: value.clone() for name, value in block.state_dict().items()}
    stages = trace_sequential(block, inputs)
    close(stages["stage3_LayerNorm"], block(inputs), atol=0, rtol=0)
    no_bias = trace_sequential(block, inputs, zero_bias=(0,))
    close(no_bias["stage0_Linear"], F.linear(inputs, block[0].weight, None), atol=0, rtol=0)
    identity = trace_sequential(block, inputs, activation="identity", skip_norm=True)
    close(identity["stage1_SiLU"], identity["stage0_Linear"], atol=0, rtol=0)
    close(identity["stage3_LayerNorm"], identity["stage2_Linear"], atol=0, rtol=0)
    assert all(torch.equal(value, block.state_dict()[name]) for name, value in parameters.items())
    return {"spectral_parseval": "passed", "entire_zero_space": "passed",
            "constant_and_empty_energy": "passed", "node_permutation": "passed",
            "block_diagonal_energy": "passed", "stage_trace_and_no_mutation": "passed"}


def model_checks(device):
    from archived_frequency.mis_v4_frequency import V4FrequencyGCN, v4_topology
    edges = edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]).to(device)
    adj = torch.sparse_coo_tensor(edges, torch.full((edges.shape[1],), 0.5, device=device), (7, 7)).to_sparse_csr()
    topology = v4_topology(edges, adj, 7)
    x = torch.randint(0, 2, (7, 4), device=device).float()
    checks = {}
    for mode in ("v4a", "v4a_message", "v4a_mlp"):
        model = V4FrequencyGCN(128, 5, mode=mode).to(device).eval()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        expected = model(x, edges, adj, topology)
        details, stages = trace_model(model, x, edges, adj, topology)
        close(stages["logits"], expected, atol=1e-6, rtol=1e-6)
        close(override_logits(model, x, edges, adj, topology), expected, atol=1e-6, rtol=1e-6)
        for branch in ("low", "high"):
            close(stages[f"{branch}_stage3_LayerNorm"], details[f"u_{branch}"], atol=0, rtol=0)
        close(override_logits(model, x, edges, adj, topology, low_scale=0, high_scale=0),
              model(x, edges, adj, topology, disable_adapter=True), atol=0, rtol=0)
        for spec in ({"zero_bias": {"high": (0,)}}, {"zero_bias": {"high": (2,)}},
                     {"activation": {"high": "identity"}}, {"skip_norm": ("high",)}, {"gate": 0.5}):
            intervention_details(model.adapter, x, adj, topology, **spec)
            values = override_logits(model, x, edges, adj, topology, **spec)
            assert torch.isfinite(values).all()
        singles = torch.cat([model(x[:, i:i + 1], edges, adj, topology) for i in range(x.shape[1])], dim=1)
        close(singles, expected, atol=1e-5, rtol=1e-5)
        assert before.keys() == model.state_dict().keys()
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
        close(model(x, edges, adj, topology), expected, atol=0, rtol=0)
        checks[mode] = "trace/logits/particles/interventions/state_dict passed"
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spectral-only", action="store_true", help="Run pure-torch checks without the original GCN dependencies")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    with torch.no_grad():
        result = {"status": "passed", "spectral": spectral_checks(), "device": args.device}
        if not args.spectral_only:
            result["models"] = model_checks(torch.device(args.device))
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
