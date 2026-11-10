"""Separate uninstrumented MIS solve timing from synchronized component benchmarks."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter

import torch

from archived_frequency.diagnose_v4_branch_replay import (MODEL_MODES, add_common_arguments, csv_file,
                                      load_group, replay, shared_noise)
from eval_v2 import ROOT, dataset_at, search, sync, write_run
from mis_v1 import adjacency
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import PAVEV3
from archived_frequency.mis_v4_frequency import lazy_walk, state_bands, v4_topology


def state_statistics(x, topology, beta=1.02):
    """The state_bands prefix, without its two lazy walks."""
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.float()
        counts = topology.binary @ x
        release = ((x == 0) & (counts == 1)).float()
        released = x * (topology.binary @ release)
        return dict(x=x, c=counts, r=released, z=torch.stack((x, counts.log1p()), -1),
                    delta_energy=(1 - 2 * x) * (beta * counts - 1))


def branch_inputs(adapter, details, topology):
    low_input = details["ppz"]
    high_input = torch.cat((details["B0"], details["B1"]), -1)
    eta = high_input.abs().mean(-1)
    x = details["x"]
    if adapter.mode == "v4a_mlp":
        low_input = details["z"]
        high_input = torch.cat((details["z"], topology.log_degree.expand_as(x).unsqueeze(-1),
                                details["r"].log1p().unsqueeze(-1)), -1)
    elif adapter.mode == "v4a_message":
        high_input = torch.cat((details["z"], details["pz"]), -1)
    elif adapter.mode != "v4a":
        raise ValueError(f"Component profiling supports the three trained adapter arms: {adapter.mode}")
    gate_input = torch.stack((x, topology.log_degree.expand_as(x), details["c"].log1p(),
                              details["r"].log1p(), details["delta_energy"].clamp(-1, 1), eta), -1)
    return low_input, high_input, gate_input


def gate_fusion(adapter, raw, low, high, gate_input):
    gates = adapter.gates(gate_input).sigmoid()
    delta_low = adapter.alpha_low * gates[..., :1] * low
    delta_high = adapter.alpha_high * gates[..., 1:] * high
    return raw + delta_low.to(raw.dtype) + delta_high.to(raw.dtype)


def component_inputs(model, state, adj, topology):
    """Check dense branch composition against one native set of sparse features."""
    raw = model.initial_project(state.unsqueeze(-1))
    values = dict(raw=raw, fused=raw, fused_max_abs=0.0)
    if not hasattr(model, "adapter"):
        return values
    adapter = model.adapter
    details = adapter.inspect(state, adj, topology)
    low_input, high_input, gate_input = branch_inputs(adapter, details, topology)
    for actual, expected in ((low_input, details["low_input"]), (high_input, details["high_input"]),
                             (gate_input, details["gate_input"])):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    low, high = adapter.low(low_input), adapter.high(high_input)
    torch.testing.assert_close(low, details["u_low"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(high, details["u_high"], atol=1e-6, rtol=1e-5)
    fused = gate_fusion(adapter, raw, low, high, gate_input)
    native_fused = raw + details["delta_low"].to(raw.dtype) + details["delta_high"].to(raw.dtype)
    torch.testing.assert_close(fused, native_fused, atol=1e-6, rtol=1e-5)
    values.update(fused=fused, details=details, low_input=low_input, high_input=high_input,
                  gate_input=gate_input, low=low, high=high,
                  fused_max_abs=(fused.float() - native_fused.float()).abs().max().item())
    return values


def forward_variation(values):
    pairs = [(values[i], values[j]) for i in range(len(values)) for j in range(i)]
    return dict(native_forward_repeats=len(values),
                native_pairwise_max_abs=max((a - b).abs().max().item() for a, b in pairs),
                native_pairwise_changed_fraction=max((a != b).float().mean().item() for a, b in pairs),
                native_probability_max_abs=max((a.sigmoid() - b.sigmoid()).abs().max().item() for a, b in pairs))


def benchmark(function, device, repeats, iterations):
    for _ in range(3):
        function()
    elapsed = []
    for repeat in range(repeats):
        sync(device)
        started = perf_counter()
        for _ in range(iterations):
            function()
        sync(device)
        elapsed.append((perf_counter() - started) / iterations)
    return elapsed


def quantile(values, fraction):
    return torch.quantile(torch.tensor(values, dtype=torch.float64), fraction).item()


def latency_summary(values):
    middle = median(values)
    return dict(mean_s=mean(values), median_s=middle, p90_s=quantile(values, .9),
                p95_s=quantile(values, .95), mad_s=median(abs(value - middle) for value in values),
                std_s=stdev(values) if len(values) > 1 else 0.0)


def build_parser():
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(precision="fp16")
    parser.add_argument("--graph-count", type=int, default=20, dest="graphs")
    parser.add_argument("--warmup", type=int, default=30, help="Full forwards per model before timing")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--component-iterations", type=int, default=20)
    parser.add_argument("--native-forward-repeats", type=int, default=3)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_MODES),
                        default=["baseline", "tuned", "mlp", "message", "f4"])
    parser.add_argument("--post", nargs="+", choices=("none", "pave"), default=["none", "pave"])
    parser.add_argument("--mode", choices=("both", "e2e", "components"), default="both")
    return parser


@torch.no_grad()
def profile(args):
    if args.graphs <= 0 or args.repeats <= 0 or args.component_iterations <= 0 or args.warmup < 0 or args.native_forward_repeats < 2:
        raise ValueError("Choose positive graph/repeat/iteration counts and a nonnegative warmup")
    torch.set_num_threads(1)
    names = list(dict.fromkeys(args.models))
    solvers, paths = load_group(args, names)
    first = solvers[names[0]]
    device = first.accelerator.device
    dataset = dataset_at(args.test_path, first.dtype, args.graphs)
    items = [dataset[index] for index in range(len(dataset))]
    graph_ids = [Path(path).stem for path in dataset.files]
    ranker = None
    pave_path = ROOT / f"outputs/v3/stats_h1_seed{args.seed}/checkpoints/best.pt"
    if "pave" in args.post:
        checkpoint = torch.load(pave_path, map_location="cpu", weights_only=False)
        ranker = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", 24)).eval().requires_grad_(False)
        ranker.load_state_dict(checkpoint["model_state"])
    args.result_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(version="v4_frequency_followup", mode="profile", status="running", config=vars(args),
                    seed=args.seed, precision=args.precision, split=args.split, checkpoints=paths,
                    graph_ids=graph_ids, started_at=datetime.now(timezone.utc).isoformat(),
                    device=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                    exclusive_gpu_required=True,
                    pave_checkpoint=str(pave_path) if ranker is not None else None,
                    e2e_protocol="Original eval_v2.search plus frozen v3 PAVE; no observer or in-step synchronization",
                    timing="Initial H2D and checkpoint/dataset loading excluded; CPU graph, topology, search, A.3, PAVE and feasibility included",
                    microbenchmark="Synchronized repeated functions on fixed same-state inputs; inclusive components overlap and must not be summed as E2E",
                    component_validation="Branch inputs, encoders and residual fusion match native inspect on the same cached sparse features, separately in FP32 and requested precision",
                    native_numerics="Independent CUDA sparse-GCN evaluations can differ; native repeat spread and decomposition-to-native differences are reported, not declared identical",
                    components_state="Common first-listed model trajectory at t=50, K20; A.3 input is its final energy archive",
                    budget_protocol="Run eval_v4_frequency.py separately with --time-budget-s 0.5; profiler does not alter deadline behavior")
    write_run(args.result_dir, metadata)
    started = perf_counter()
    end_rows, component_rows, equivalence_rows = [], [], []
    try:
        warm_graph = items[0].to(device)
        warm_adj = adjacency(warm_graph)
        initial, _, _ = shared_noise(warm_graph.num_nodes, 20, 100, device, first.dtype, args.search_seed)
        for name in names:
            solver = solvers[name]
            topology = solver.policy_topology(warm_graph, warm_adj)
            for _ in range(args.warmup):
                solver.flip_logits(initial, warm_graph, warm_adj, topology)
            cpu_graph = SearchGraph(warm_graph.num_nodes, warm_graph.edge_index, seed=args.search_seed)
            selected, _ = search(solver, warm_graph, cpu_graph, args.search_seed)
            if ranker is not None:
                platform_search(cpu_graph, selected, choose=ranker.choose, horizon=1,
                                initial_rounds=3, outer_rounds=8, candidate_cap=32,
                                improving_cap=48, repair_rounds=4)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if args.mode != "components":
            for index, item in enumerate(items):
                edges = item.edge_index.cpu()
                graph = item.to(device)
                for repeat in range(args.repeats):
                    offset = (index + repeat) % len(names)
                    order = names[offset:] + names[:offset]
                    for position, name in enumerate(order):
                        solver = solvers[name]
                        sync(device)
                        tick = perf_counter()
                        cpu_graph = SearchGraph(graph.num_nodes, edges, seed=args.search_seed + index)
                        selected, steps = search(solver, graph, cpu_graph, args.search_seed + index)
                        sync(device)
                        shared_seconds = perf_counter() - tick
                        for post in dict.fromkeys(args.post):
                            post_start = perf_counter()
                            final = selected
                            if post == "pave":
                                final = platform_search(cpu_graph, selected, choose=ranker.choose, horizon=1,
                                                        initial_rounds=3, outer_rounds=8, candidate_cap=32,
                                                        improving_cap=48, repair_rounds=4).best
                            if not cpu_graph.feasible(final) or (graph.num_nodes and not final):
                                raise ValueError(f"Invalid E2E independent set: {graph_ids[index]}/{name}/{post}")
                            post_seconds = perf_counter() - post_start
                            end_rows.append(dict(graph_id=graph_ids[index], graph_index=index, seed=args.seed,
                                                 precision=args.precision, model=name, repeat=repeat, order=position,
                                                 post=post, base_size=selected.bit_count(), final_size=final.bit_count(),
                                                 search_steps=steps, solve_time_s=shared_seconds + post_seconds,
                                                 search_time_s=shared_seconds, post_time_s=post_seconds,
                                                 feasible=1, nodes=graph.num_nodes, directed_edges=graph.edge_index.shape[1]))
                print(f"E2E {index + 1}/{len(items)} graphs completed", flush=True)
            csv_file(args.result_dir / "end_to_end.csv", end_rows)

        if args.mode != "e2e":
            for index, item in enumerate(items):
                edges = item.edge_index.cpu()
                graph = item.to(device)
                adj = adjacency(graph)
                teacher_topology = first.policy_topology(graph, adj)
                initial, uniforms, _ = shared_noise(graph.num_nodes, 20, 100, device, first.dtype, args.search_seed + index)
                common = replay(first.problem, adj, graph.b, initial, uniforms,
                                lambda x: first.flip_logits(x, graph, adj, teacher_topology), phases=(50,), capture=True)
                state, best = common["states"][50], common["best_states"]
                cpu_graph = SearchGraph(graph.num_nodes, edges, seed=args.search_seed + index)
                ft = v4_topology(graph.edge_index, adj, graph.num_nodes)
                for name in names[index % len(names):] + names[:index % len(names)]:
                    solver = solvers[name]
                    model = solver.accelerator.unwrap_model(solver.model)
                    topology = solver.policy_topology(graph, adj)
                    normalized = topology.normalized if hasattr(topology, "normalized") else topology
                    with torch.autocast(device_type=device.type, enabled=False):
                        fp32 = component_inputs(model, state.float(), adj.float(), topology)
                        fp32_decomposed = model.project_hidden(fp32["fused"], normalized).squeeze(-1).float()
                        if hasattr(model, "adapter"):
                            fp32_native = type(model).forward(model, state.float(), graph.edge_index, adj.float(), topology).float()
                        else:
                            fp32_native = type(model).forward(model, state.float().unsqueeze(-1), graph.edge_index, normalized).squeeze(-1).float()
                    with solver.accelerator.autocast():
                        prepared = component_inputs(model, state, adj, topology)
                        raw, fused = prepared["raw"], prepared["fused"]
                        if hasattr(model, "adapter"):
                            adapter = model.adapter
                            details = prepared["details"]
                            low_input, high_input, gate_input = (prepared[key] for key in ("low_input", "high_input", "gate_input"))
                            low, high = prepared["low"], prepared["high"]
                        expected = model.project_hidden(fused, normalized).squeeze(-1).float()
                    native_values = [solver.flip_logits(state, graph, adj, topology).detach().clone()
                                     for _ in range(args.native_forward_repeats)]
                    actual = native_values[0]
                    equivalence_rows.append(dict(
                        graph_id=graph_ids[index], graph_index=index, seed=args.seed, precision=args.precision, model=name,
                        fp32_cached_fusion_max_abs=fp32["fused_max_abs"],
                        precision_cached_fusion_max_abs=prepared["fused_max_abs"],
                        fp32_decomposition_vs_native_max_abs=(fp32_decomposed - fp32_native).abs().max().item(),
                        decomposition_vs_native_max_abs=max((expected - value).abs().max().item() for value in native_values),
                        **forward_variation(native_values)))

                    def amp(function):
                        def call():
                            with solver.accelerator.autocast():
                                return function()
                        return call

                    generator = torch.Generator(device=device).manual_seed(args.search_seed + index)
                    current_energy, _ = solver.problem.energy_func(adj, graph.b, state)
                    probability = actual.sigmoid()

                    def transition():
                        uniform = torch.rand(state.shape, dtype=state.dtype, device=device, generator=generator)
                        after = torch.where(uniform < probability, 1 - state, state)
                        energy, _ = solver.problem.energy_func(adj, graph.b, after)
                        better = energy < current_energy
                        archive = torch.where(better.unsqueeze(0), after, state)
                        return archive, torch.where(better, energy, current_energy)

                    components = {
                        "cpu_graph": (lambda: SearchGraph(graph.num_nodes, edges, seed=args.search_seed + index), "per_graph"),
                        "energy_adjacency": (lambda: adjacency(graph), "per_graph"),
                        "policy_topology": (lambda: solver.policy_topology(graph, adj), "per_graph"),
                        "initial_embedding": (amp(lambda: model.initial_project(state.unsqueeze(-1))), "per_step"),
                        "gcn_backbone": (amp(lambda: model.project_hidden(fused, normalized)), "per_step"),
                        "full_logits": (lambda: solver.flip_logits(state, graph, adj, topology), "per_step"),
                        "bernoulli_energy_archive": (transition, "per_step"),
                        "a3_decode": (lambda: solver.problem.decode_gradient_flip(best, adj, graph.b, return_solutions=True), "per_graph"),
                    }
                    if hasattr(model, "adapter"):
                        def walks():
                            pz = lazy_walk(details["z"], ft)
                            return lazy_walk(pz, ft)
                        components.update(
                            state_statistics=(lambda: state_statistics(state, ft), "per_step"),
                            lazy_walk_twice=(walks, "per_step"),
                            state_bands_total=(lambda: state_bands(state, ft), "per_step"),
                            branch_inputs=(lambda: branch_inputs(adapter, details, ft), "per_step"),
                            low_encoder=(amp(lambda: adapter.low(low_input)), "per_step"),
                            high_encoder=(amp(lambda: adapter.high(high_input)), "per_step"),
                            gate_fusion=(amp(lambda: gate_fusion(adapter, raw, low, high, gate_input)), "per_step"),
                            adapter_total=(amp(lambda: adapter.inspect(state, adj, ft)), "per_step"),
                        )
                    if ranker is not None:
                        components["pave"] = (lambda: platform_search(cpu_graph, common["selected"], choose=ranker.choose,
                                                                     horizon=1, initial_rounds=3, outer_rounds=8,
                                                                     candidate_cap=32, improving_cap=48, repair_rounds=4), "per_graph")
                    for component, (function, unit) in components.items():
                        iterations = 1 if component in ("pave", "a3_decode") else args.component_iterations
                        values = benchmark(function, device, args.repeats, iterations)
                        summary = latency_summary(values)
                        for repeat, elapsed in enumerate(values):
                            component_rows.append(dict(graph_id=graph_ids[index], graph_index=index, seed=args.seed,
                                                       precision=args.precision, model=name, component=component,
                                                       unit=unit, repeat=repeat, iterations=iterations, time_s=elapsed,
                                                       equivalent_100_steps_s=elapsed * 100 if unit == "per_step" else None,
                                                       nodes=graph.num_nodes, directed_edges=graph.edge_index.shape[1], **summary))
                print(f"Components {index + 1}/{len(items)} graphs completed", flush=True)
            csv_file(args.result_dir / "components.csv", component_rows)
            csv_file(args.result_dir / "component_equivalence.csv", equivalence_rows)
        summaries = []
        for name in names:
            for post in args.post:
                values = [row["solve_time_s"] for row in end_rows if row["model"] == name and row["post"] == post]
                if values:
                    summaries.append(dict(model=name, post=post, **latency_summary(values)))
        metadata.update(status="completed", samples=len(items), e2e_summary=summaries,
                        peak_memory_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0,
                        wall_time_s=perf_counter() - started, finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        metadata.update(status="failed", error=str(error), wall_time_s=perf_counter() - started)
        write_run(args.result_dir, metadata)
        raise
    write_run(args.result_dir, metadata)


if __name__ == "__main__":
    profile(build_parser().parse_args())
