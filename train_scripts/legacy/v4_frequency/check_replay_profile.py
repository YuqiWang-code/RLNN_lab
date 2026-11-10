"""Check shared-random replay and profiler decomposition on a small MIS graph."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))
from archived_frequency.diagnose_v4_branch_replay import policy_logits, replay, same_state_rows, shared_noise
from eval_v2 import search
from archived_frequency.frequency_diag_utils import override_logits
from mis_v3_search import SearchGraph
from archived_frequency.mis_v4_frequency import V4FrequencyGCN, state_bands, v4_topology
from problems.mis import Problem
from archived_frequency.profile_v4_frequency import branch_inputs, gate_fusion, state_statistics
from solvers import RLNN


@torch.no_grad()
def check(device, precision):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    pairs = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    edges = torch.tensor(pairs + [(j, i) for i, j in pairs], device=device).T.contiguous()
    adjacency = torch.sparse_coo_tensor(edges, torch.full((edges.shape[1],), .5, device=device, dtype=dtype),
                                        (7, 7)).to_sparse_csr()
    graph = SimpleNamespace(num_nodes=7, edge_index=edges, edge_weight=adjacency.values(),
                            b=-torch.ones(7, 1, device=device, dtype=dtype))
    topology = v4_topology(edges, adjacency, 7)
    model = V4FrequencyGCN(16, 2, mode="v4a").to(device).eval().requires_grad_(False)
    context = lambda: torch.autocast(device_type=device.type, dtype=torch.float16) if precision == "fp16" else nullcontext()
    accelerator = SimpleNamespace(device=device, autocast=context, unwrap_model=lambda module: module)
    solver = SimpleNamespace(model=model, problem=Problem(1.02), dtype=dtype, num_k=20, num_t=100,
                              accelerator=accelerator, policy_topology=lambda *_: topology)

    def logits(state, *_):
        with context():
            return model(state, edges, adjacency, topology).float()

    solver.flip_logits = logits
    solver.sample_states = lambda *args, **kwargs: RLNN.sample_states(solver, *args, **kwargs)
    initial, uniforms, probe = shared_noise(7, 20, 100, device, dtype, 1017)
    generator = torch.Generator(device=device).manual_seed(1017)
    torch.testing.assert_close(initial, torch.randint(0, 2, (7, 20), device=device, dtype=dtype, generator=generator), atol=0, rtol=0)
    for uniform in uniforms:
        torch.testing.assert_close(uniform, torch.rand((7, 20), device=device, dtype=dtype, generator=generator), atol=0, rtol=0)
    torch.testing.assert_close(probe, torch.rand((7, 20), device=device, dtype=dtype, generator=generator), atol=0, rtol=0)
    result = replay(solver.problem, adjacency, graph.b, initial, uniforms, logits, phases=range(101), capture=True)
    repeated = replay(solver.problem, adjacency, graph.b, initial, uniforms, logits, phases=range(101), capture=True)
    generator = torch.Generator(device=device).manual_seed(1017)
    stream = solver.sample_states(graph, adjacency, generator=generator, normalized=topology)
    for step, state in enumerate(stream):
        torch.testing.assert_close(result["states"][step], state, atol=0, rtol=0)
        torch.testing.assert_close(repeated["states"][step], state, atol=0, rtol=0)
    cpu_graph = SearchGraph(7, edges.cpu(), seed=1017)
    selected, steps = search(solver, graph, cpu_graph, 1017)
    assert steps == 100 and selected == result["selected"] == repeated["selected"]
    assert cpu_graph.feasible(selected) and selected
    tolerance = 1e-5 if precision == "fp32" else 2e-3
    before = {name: value.clone() for name, value in model.state_dict().items()}
    full = policy_logits(solver, graph, adjacency, topology, initial)
    with context():
        no_adapter = model(initial, edges, adjacency, topology, disable_adapter=True).float()
    intervention = policy_logits(solver, graph, adjacency, topology, initial, dict(low_scale=0., high_scale=0.))
    torch.testing.assert_close(intervention, no_adapter, atol=tolerance, rtol=tolerance)
    policy_logits(solver, graph, adjacency, topology, initial, dict(zero_bias={"high": (0,)}, activation={"high": "identity"}))
    torch.testing.assert_close(full, policy_logits(solver, graph, adjacency, topology, initial), atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    details = state_bands(initial, topology)
    stats = state_statistics(initial, topology)
    for name in stats:
        torch.testing.assert_close(stats[name], details[name], atol=0, rtol=0)
    for mode in ("v4a", "v4a_mlp", "v4a_message"):
        candidate = V4FrequencyGCN(16, 2, mode=mode).to(device).eval()
        with context():
            native = candidate.adapter.inspect(initial, adjacency, topology)
            low_input, high_input, gate_input = branch_inputs(candidate.adapter, details, topology)
            for actual, expected in ((low_input, native["low_input"]), (high_input, native["high_input"]), (gate_input, native["gate_input"])):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            raw = candidate.initial_project(initial.unsqueeze(-1))
            fused = gate_fusion(candidate.adapter, raw, candidate.adapter.low(low_input), candidate.adapter.high(high_input), gate_input)
            decomposed = candidate.project_hidden(fused, topology.normalized).squeeze(-1)
            native_logits = candidate(initial, edges, adjacency, topology)
        torch.testing.assert_close(decomposed, native_logits, atol=tolerance, rtol=tolerance)
    rows = same_state_rows(initial, details["c"], details["r"], {"full": full, "no_adapter": intervention}, uniforms[0], binary=topology.binary)
    reference_rows = [row for row in rows if row["variant"] == "full" and row["category"] == "all"]
    assert len(reference_rows) == 20
    assert all(row["top5_jaccard"] == 1 and row["sampled_action_disagreement"] == 0 for row in reference_rows)
    q = initial.float() + (1 - 2 * initial.float()) * full.float().sigmoid()
    exact_expected = (q * (1.02 * (topology.binary @ q) / 2 - 1)).sum(0)
    current = (initial.float() * (1.02 * details["c"] / 2 - 1)).sum(0)
    torch.testing.assert_close(torch.tensor([row["expected_joint_delta_energy"] for row in reference_rows], device=device),
                               exact_expected - current, atol=1e-4, rtol=1e-5)
    return dict(precision=precision, size=selected.bit_count(), steps=steps, shared_stream=True,
                matches_original_search=True, interventions_nonmutating=True, component_decomposition=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    device = torch.device(args.device)
    result = dict(device=args.device, checks=[check(device, precision) for precision in
                                             (("fp32", "fp16") if args.device == "cuda" else ("fp32",))])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
