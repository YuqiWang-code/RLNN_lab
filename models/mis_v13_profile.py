"""R2-E3 (t17): kernel-level profile of the A0 one-step path.

Measurement only.  Nothing here touches the model, the weights, ``K``,
``decode_every`` or any numerical path: the script builds the *same* engine the
A0 gate uses (``--dense --mixed_precision fp16 --num_k 200 --chunk 10``),
replays the captured block, and reads the CUDA kernel table out of
``torch.profiler``.  It never writes into a run directory, only into
``--out``.

What it answers, for the "is there 2x of step throughput left" call:

(A) where the ~0.87 ms/step goes: top kernels (share of device time,
    calls per step, us per call) for the captured block and for the eager
    path -- the eager path issues the same arithmetic without the graph, so the
    difference between the two is the launch/scheduling share;
    plus the achieved GFLOP/s of the 264x264x200 GEMM against the card's fp16
    tensor peak, and a same-shape ``torch.matmul`` micro-benchmark, which is
    the cheapest available estimate of what a better kernel could reach at
    this size.

(B) the ceiling of the three speed-up families, each backed by a measured
    number: fewer launches (captured vs eager gap), different math (same-shape
    gemm at fp16/bf16/fp32 -- the dtype ladder the kernels actually deliver),
    reuse across graphs (capture cost amortised over the 1000 graphs).

(C) whether >=2x step throughput is reachable without changing the model.

    CUDA_VISIBLE_DEVICES=1 python models/mis_v13_profile.py \
        --out outputs/v11/r2/e3_profile
"""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode

from eval_v2 import dataset_at, load_policy, sync
from eval_v8_padded import build_parser, shape_of
from mis_v12_parallel import engine_for

# Board spec sheets, dense (no sparsity) fp16/bf16 tensor throughput, TFLOPS.
# Only used for the utilisation ratio; the ratio is a sanity check, not a claim.
PEAK_FP16_TFLOPS = {
    "NVIDIA GeForce RTX 5090": 209.5,
    "NVIDIA GeForce RTX 4090": 165.2,
    "NVIDIA GeForce RTX 3090": 71.0,
    "NVIDIA A100-SXM4-40GB": 312.0,
    "NVIDIA H100 PCIe": 756.0,
}


def parse(argv):
    parser = build_parser()
    parser.add_argument("--graph_index", type=int, default=0,
                        help="dataset index to profile (dataset order, no cherry-picking)")
    parser.add_argument("--blocks", type=int, default=20,
                        help="captured blocks to profile (one block = --chunk steps; 20 = 200 steps)")
    parser.add_argument("--eager_steps", type=int, default=50,
                        help="eager steps to profile")
    parser.add_argument("--warm_blocks", type=int, default=5,
                        help="warm-up blocks before the captured profile")
    parser.add_argument("--top", type=int, default=15, help="kernels to keep per table")
    parser.add_argument("--out", default="outputs/v11/r2/e3_profile",
                        help="directory for kernels.json")
    # The A0 gate configuration, pinned here so the profile cannot silently
    # measure a different path: fp16 + dense + K=200 + chunk 10 (the shell
    # script's ``COMMON`` flags, see ``train_scripts/v11/a0_gate0.sh``).
    parser.set_defaults(budgets=[1.0], seeds=[1017], repeats=1, shards=1, shard=0,
                        decoders=["captured"], num_k=200, chunk=10, dense=True,
                        mixed_precision="fp16", reference_seed=0,
                        save_dir=Path("outputs/v2/none_seed0/checkpoints"),
                        result_dir=Path("tmp/e3_profile_unused"))
    # ``--result_dir`` / ``--reference_seed`` are argparse-required in the eval
    # parser; this script writes no run directory, so their defaults suffice.
    for action in parser._actions:
        if action.dest in {"result_dir", "reference_seed"}:
            action.required = False
    return parser.parse_args(argv)


def device_report(device):
    properties = torch.cuda.get_device_properties(device)
    name = properties.name
    peak = PEAK_FP16_TFLOPS.get(name)
    return dict(name=name, sm_count=int(properties.multi_processor_count),
                total_gib=round(properties.total_memory / 1024 ** 3, 3),
                capability=f"{properties.major}.{properties.minor}",
                peak_fp16_tflops=peak,
                torch=torch.__version__, cuda=torch.version.cuda)


def analytic_gemm_flops(model, nodes, k):
    """FLOPs of the graph matmuls only (``adj @ support``), from module shapes.

    Every ``GCNConv`` in this family multiplies ``(nodes, nodes)`` by
    ``(nodes*k, hidden)``, i.e. ``2*nodes*nodes*k*hidden`` FLOPs per step.
    """
    total, layers = 0, []
    for module in model.modules():
        if module.__class__.__name__ != "GCNConv":
            continue
        hidden = module.out_features
        flops = 2 * nodes * nodes * k * hidden
        total += flops
        layers.append(dict(in_features=module.in_features, out_features=hidden,
                           flops=flops))
    return total, layers


def model_report(model):
    linear = [(m.in_features, m.out_features) for m in model.modules()
              if isinstance(m, torch.nn.Linear)]
    return dict(parameters=int(sum(p.numel() for p in model.parameters())),
                linears=len(linear), linear_shapes=linear[:6],
                convs=sum(1 for m in model.modules()
                          if m.__class__.__name__ == "GCNConv"))


def group_of(name):
    """Which part of the step a kernel belongs to (name heuristics)."""
    lowered = name.lower()
    if any(tag in lowered for tag in ("gemm", "cutlass", "csrmm", "spmm", "gemv", "gemmk1")):
        return "gemm"
    if any(tag in lowered for tag in ("reduce", "softmax")):
        return "reduce"
    if any(tag in lowered for tag in ("rand", "philox", "distribution")):
        return "sample"
    if any(tag in lowered for tag in ("elementwise", "memcpy", "copy", "fill", "cast")):
        return "pointwise"
    return "other"


def group_rows(rows):
    groups = {}
    for row in rows:
        key = group_of(row["name"])
        bucket = groups.setdefault(key, dict(us_per_step=0.0, kernels=0, calls_per_step=0.0))
        bucket["us_per_step"] += row["device_us_per_step"]
        bucket["kernels"] += 1
        bucket["calls_per_step"] += row["count_per_step"]
    for bucket in groups.values():
        bucket["us_per_step"] = round(bucket["us_per_step"], 2)
        bucket["calls_per_step"] = round(bucket["calls_per_step"], 2)
    return groups


def kernel_rows(profiler, steps, top):
    """Top device-time kernels, normalised per step."""
    rows = []
    for event in profiler.key_averages():
        device_us = float(getattr(event, "self_device_time_total", 0.0))
        if device_us <= 0.0:
            continue
        cpu_us = float(getattr(event, "self_cpu_time_total", 0.0))
        calls = int(event.count)
        rows.append(dict(name=event.key, device_us_per_step=device_us / steps,
                         device_us_per_call=device_us / calls,
                         count_per_step=calls / steps,
                         cpu_us_per_step=cpu_us / steps))
    total = sum(row["device_us_per_step"] for row in rows)
    for row in rows:
        row["share"] = row["device_us_per_step"] / total if total else 0.0
    rows.sort(key=lambda row: row["device_us_per_step"], reverse=True)
    return rows[:top], rows, total


def timed_blocks(engine, blocks, chunk):
    """Wall ms/step of the captured block (A0 semantics: one sync per block)."""
    engine.graph_exec.replay()
    sync(engine.device)
    times = []
    for _ in range(blocks):
        started = perf_counter()
        engine.graph_exec.replay()
        torch.cuda.synchronize(engine.device)
        times.append((perf_counter() - started) * 1000.0 / chunk)
    return times


def micro_gemm(nodes, k, hidden, device, dtypes, iters=50):
    """Same-shape ``adj @ support`` through cuBLAS: the kernel ceiling at this size."""
    adj = torch.rand((nodes, nodes), device=device)
    # ``GCNConv`` flattens ``(nodes, k, hidden)`` to ``(nodes, k*hidden)`` before
    # the adjacency matmul, so the contract is (nodes, nodes) x (nodes, k*hidden).
    support = torch.rand((nodes, k * hidden), device=device)
    results = {}
    for dtype in dtypes:
        left = adj.to(dtype)
        right = support.to(dtype)
        for _ in range(5):
            torch.matmul(left, right)
        sync(device)
        started = perf_counter()
        for _ in range(iters):
            torch.matmul(left, right)
        sync(device)
        seconds = perf_counter() - started
        flops = 2 * nodes * nodes * k * hidden
        results[str(dtype).replace("torch.", "")] = dict(
            ms=seconds * 1000.0 / iters,
            gflops=flops * iters / seconds / 1e9)
    return results


def main():
    args = parse(None)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("this profile needs CUDA (a CPU run measures nothing)")

    solver = load_policy(args)
    solver.model.requires_grad_(False)
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    n_max, capacity = shape_of(dataset, args.padded_shapes)
    graph = dataset[args.graph_index]
    nodes = int(graph.num_nodes)
    print(f"graph {args.graph_index}: nodes {nodes}, padded {n_max}/{capacity}", flush=True)

    # --- capture cost + engine (identical flags to the A0 gate) ------------
    entered = perf_counter()
    engine = engine_for(solver, 1, n_max, capacity, args, device)
    capture_seconds = perf_counter() - entered
    engine.load(graph.to(device), graph.edge_index, args.seeds[0])
    engine.run(perf_counter() + 0.05, lambda states: None)
    sync(device)
    solver.num_t = 20000
    chunk = engine.chunk
    steps_captured = args.blocks * chunk
    print(f"capture {capture_seconds:.3f} s, chunk {chunk}, "
          f"num_k {engine.num_k}, dtype {engine.dtype}, dense {engine.use_dense}", flush=True)

    # --- wall clock (outside the profiler) --------------------------------
    for _ in range(args.warm_blocks):
        engine.graph_exec.replay()
    sync(device)
    wall_captured = timed_blocks(engine, args.blocks, chunk)
    wall_per_step = median(wall_captured)

    # --- (A) captured kernel table ---------------------------------------
    # CUPTI registers itself on the first profiled region and that region comes
    # back empty, so burn one block on a throw-away profile first.
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as warm:
        engine.graph_exec.replay()
        torch.cuda.synchronize(device)
    warm_events = sum(1 for _ in warm.key_averages())
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
        for _ in range(args.blocks):
            engine.graph_exec.replay()
        torch.cuda.synchronize(device)
    rows_captured, all_captured, kernel_captured = kernel_rows(captured, steps_captured, args.top)
    print(f"profiler: warm-up region {warm_events} rows, captured region "
          f"{len(list(captured.key_averages()))} rows", flush=True)

    # --- (A) eager kernel table ------------------------------------------
    # ``_body_one`` is the same arithmetic with ``chunk = 1`` and no capture:
    # every step goes through the host launch path, which is exactly the
    # launch/scheduling share the graph removes.
    for _ in range(5):
        engine._body_one()
    sync(device)
    started = perf_counter()
    for _ in range(args.eager_steps):
        engine._body_one()
    sync(device)
    wall_eager = (perf_counter() - started) * 1000.0 / args.eager_steps
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as eager:
        for _ in range(args.eager_steps):
            engine._body_one()
        torch.cuda.synchronize(device)
    rows_eager, all_eager, kernel_eager = kernel_rows(eager, args.eager_steps, args.top)

    # --- FLOPs per step ---------------------------------------------------
    with FlopCounterMode(display=False) as counter:
        engine._body_one()
    flops_counter = int(counter.get_total_flops())
    gemm_flops, gemm_layers = analytic_gemm_flops(solver.model, n_max, engine.num_k)

    def gemm_share(rows):
        return sum(row["device_us_per_step"] for row in rows
                   if "gemm" in row["name"].lower() or "cutlass" in row["name"].lower()
                   or "matmul" in row["name"].lower())

    captured_gemm_us = gemm_share(all_captured)
    eager_gemm_us = gemm_share(all_eager)
    achieved = dict(
        flops_per_step=flops_counter,
        gemm_flops_per_step=gemm_flops,
        gemm_layers=len(gemm_layers),
        gemm_us_per_step_captured=captured_gemm_us,
        gemm_us_per_step_eager=eager_gemm_us,
        gflops_gemm_captured=(gemm_flops / (captured_gemm_us * 1e-6) / 1e9
                              if captured_gemm_us > 0 else None),
        gflops_gemm_eager=(gemm_flops / (eager_gemm_us * 1e-6) / 1e9
                           if eager_gemm_us > 0 else None),
        gflops_step_captured=flops_counter / (wall_per_step * 1e-3) / 1e9,
    )

    # --- (B) kernel ceiling at this shape --------------------------------
    hidden = gemm_layers[0]["out_features"] if gemm_layers else 100
    micro = micro_gemm(n_max, engine.num_k, hidden, device,
                       (torch.float16, torch.bfloat16, torch.float32))
    groups_captured = group_rows(all_captured)
    groups_eager = group_rows(all_eager)
    kernels_per_step = sum(row["count_per_step"] for row in all_captured)
    elementwise = [row["device_us_per_call"] for row in all_captured
                   if group_of(row["name"]) in ("pointwise", "sample", "reduce")]
    min_kernel_us = min(elementwise) if elementwise else None
    # A speed-up ceiling, spelled out: every gemm at the micro-benchmark rate,
    # every pointwise/reduce kernel at the *measured* minimum kernel latency
    # (i.e. perfect fusion into one launch each, zero DRAM cost).
    gemm_floor_us = gemm_flops / (micro["float16"]["gflops"] * 1e3)
    latency_floor_us = kernels_per_step * min_kernel_us if min_kernel_us else None
    other_us = sum(bucket["us_per_step"] for key, bucket in groups_captured.items()
                   if key not in ("gemm", "pointwise", "sample", "reduce"))
    floor = dict(kernels_per_step=round(kernels_per_step, 2),
                 min_elementwise_kernel_us=round(min_kernel_us, 3) if min_kernel_us else None,
                 gemm_floor_us=round(gemm_floor_us, 1),
                 latency_floor_us=round(latency_floor_us, 1) if latency_floor_us else None,
                 other_us=round(other_us, 1))
    if latency_floor_us:
        best_case_us = gemm_floor_us + latency_floor_us + other_us
        floor["best_case_ms_per_step"] = round(best_case_us / 1000.0, 4)
        floor["speedup_ceiling"] = round(wall_per_step / (best_case_us / 1000.0), 2)
    report = dict(
        graph_index=args.graph_index, nodes=nodes, padded=(n_max, capacity),
        num_k=int(engine.num_k), chunk=int(chunk), dtype=str(engine.dtype),
        dense=bool(engine.use_dense),
        capture_seconds=round(capture_seconds, 4),
        steps_captured=steps_captured, steps_eager=args.eager_steps,
        wall_ms_per_step_captured=wall_per_step,
        wall_ms_per_step_captured_all=wall_captured,
        wall_ms_per_step_eager=wall_eager,
        kernel_ms_per_step_captured=kernel_captured / 1000.0,
        kernel_ms_per_step_eager=kernel_eager / 1000.0,
        gap_ms_per_step_captured=wall_per_step - kernel_captured / 1000.0,
        launch_gap_share=(wall_per_step - kernel_captured / 1000.0) / wall_per_step,
        eager_over_captured=wall_eager / wall_per_step,
        kernels_captured=rows_captured, kernels_eager=rows_eager,
        groups_captured=groups_captured, groups_eager=groups_eager, floor=floor,
        achieved=achieved, micro_gemm=micro, micro_shape=(n_max, engine.num_k, hidden),
        model=model_report(solver.model), device=device_report(device),
        builder="models/mis_v13_profile.py",
    )
    peak_gflops = (device_report(device)["peak_fp16_tflops"] or 0.0) * 1000.0
    report["achieved"]["peak_fraction"] = (
        (achieved["gflops_gemm_captured"] or 0.0) / peak_gflops if peak_gflops else None)
    report["achieved"]["gflops_step_peak_fraction"] = (
        achieved["gflops_step_captured"] / peak_gflops if peak_gflops else None)
    report["floor"]["speedup_ceiling"] = floor.get("speedup_ceiling")
    path = out / "kernels.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\nwall {wall_per_step:.4f} ms/step (captured), eager {wall_eager:.4f}")
    print(f"device busy {report['kernel_ms_per_step_captured']:.4f} ms/step "
          f"=> gap {report['gap_ms_per_step_captured']:.4f} "
          f"({report['launch_gap_share']:.1%})")
    print(f"flops/step {flops_counter/1e6:.2f} MFLOP, gemm {gemm_flops/1e6:.2f} MFLOP "
          f"in {captured_gemm_us:.1f} us => "
          f"{achieved['gflops_gemm_captured'] or float('nan'):.1f} GFLOP/s "
          f"({report['achieved']['peak_fraction'] or 0.0:.1%} of peak)")
    print("\ntop kernels (captured):")
    for row in rows_captured[:args.top]:
        print(f"  {row['share']:6.1%} {row['count_per_step']:6.2f}/step "
              f"{row['device_us_per_call']:8.1f} us/call  {row['name'][:70]}")
    print("\nper group (captured):", json.dumps(groups_captured))
    print("floor:", json.dumps(floor))
    print("\nmicro gemm (same shape):", json.dumps(micro))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
