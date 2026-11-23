"""P1 probe: can pointwise fusion cut the per-step cost without touching numbers?

Three variants of the same frozen policy / same padded engine:

``captured``       what A0 runs today: ``_body`` (10 steps) inside one CUDA graph
``compiled_model`` ``torch.compile(model)`` with our own CUDA-graph capture on top
``compiled_body``  ``torch.compile(_body)`` in ``reduce-overhead`` mode (no own graph)

For each variant the probe reports ms/step (median of per-block walls, one sync per
block) and a numeric check: 130 steps from a fixed ``load()`` must land on the same
decoded ``search_size`` and the same best energy as ``captured``.

Read-only: nothing is written except ``--out`` (a JSON).  No weights, widths, K or
decode cadence are touched.
"""

import json
import statistics
import time
from pathlib import Path

import torch

from mis_v12_parallel import engine_for, parsed, setup
from mis_v13_profile import timed_blocks
from mis_v1 import adjacency  # noqa: F401  (kept for parity with the other probes)


def sizes_of(engine, nodes):
    """Decoded IS size of the engine archive (same A.3 path the eval uses)."""
    return int(engine.decode_best(engine.best_states, count_nodes=nodes).bit_count())


def run_blocks(engine, blocks, kind):
    walls = []
    for _ in range(blocks):
        start = time.perf_counter()
        if kind == "compiled_body":
            engine._body()
        else:
            engine.graph_exec.replay()
        torch.cuda.synchronize(engine.device)
        walls.append((time.perf_counter() - start) * 1000.0)
    return walls


def prepare(engine, kind):
    """Build whatever capture/compile the variant needs and warm it up."""
    if kind == "compiled_body":
        engine.body_compiled = torch.compile(engine._body, mode="reduce-overhead",
                                             fullgraph=False, dynamic=False)
        engine.__dict__["_body"] = engine.body_compiled
        engine._body()  # compile + warm
        return engine
    if kind == "compiled_model":
        engine.model = torch.compile(engine.model, mode="default", fullgraph=False,
                                     dynamic=False)
        engine._body()  # trigger compile while it is not yet captured
        engine._build()
        engine.graph_exec.replay()
        return engine
    engine.graph_exec.replay()
    return engine


def main():
    args = parsed(None)
    device = torch.device("cuda:0")
    solver, _ranker, dataset = setup(args)
    graph = dataset[0]
    nodes = int(graph.num_nodes)
    edges_index = graph.edge_index
    from eval_v8_padded import shape_of
    n_max, capacity = shape_of(dataset, args.padded_shapes)
    graph = graph.to(device)
    engine = engine_for(solver, 1, n_max, capacity, args, device)
    engine.load(graph, edges_index, int(args.seeds[0]))
    blocks = max(1, int(getattr(args, "blocks", 20)))

    results = {}
    reference = None
    for kind in ("captured", "compiled_model", "compiled_body"):
        engine.load(graph, edges_index, int(args.seeds[0]))
        engine = prepare(engine, kind)
        engine.load(graph, edges_index, int(args.seeds[0]))
        walls = run_blocks(engine, blocks, kind)
        size = sizes_of(engine, nodes)
        energy = float(engine.best_energy.max().item())
        results[kind] = {
            "ms_per_step": statistics.median(walls) / engine.chunk,
            "ms_per_block": statistics.median(walls),
            "walls_ms": [round(value, 3) for value in walls],
            "search_size": size,
            "best_energy_max": energy,
        }
        if reference is None:
            reference = (size, energy)
    for kind, entry in results.items():
        entry["size_matches"] = entry["search_size"] == reference[0]
        entry["energy_matches"] = abs(entry["best_energy_max"] - reference[1]) < 1e-6
    base = results["captured"]["ms_per_step"]
    for kind, entry in results.items():
        entry["speedup"] = base / entry["ms_per_step"]
    report = {"device": torch.cuda.get_device_name(device), "nodes": nodes,
              "n_max": n_max, "capacity": capacity, "blocks": blocks,
              "chunk": engine.chunk, "variants": results,
              "gate_1p4x": {kind: entry["speedup"] >= 1.4
                            for kind, entry in results.items()}}
    out = Path(args.out) if args.out else Path("outputs/v11/r2/e3_profile/p1_fusion.json")
    if not str(out).endswith(".json"):
        out = out / "p1_fusion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'variant':16s} {'ms/step':>8s} {'speedup':>8s} {'size':>5s} {'eq':>4s}")
    for kind, entry in results.items():
        print(f"{kind:16s} {entry['ms_per_step']:8.3f} {entry['speedup']:8.2f} "
              f"{entry['search_size']:5d} {str(entry['size_matches']):>4s}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
