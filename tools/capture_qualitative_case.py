"""Record the first author test graph and seed at ten-step intervals."""

import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from eval_v8_padded import PaddedSearch, adjacency, build_parser, dataset_at, decode_top_best, load_policy


def main():
    parser = build_parser()
    parser.add_argument("--case_file", type=Path, required=True)
    parser.add_argument("--shape_summary", type=Path, required=True)
    args = parser.parse_args()
    shape = json.loads(args.shape_summary.read_text())["padded"]
    solver = load_policy(args)
    solver.model.requires_grad_(False)
    solver.model = torch.compile(solver.model, mode="default", fullgraph=False, dynamic=False)
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples, official=True)
    graph = dataset[0]
    edges = graph.edge_index
    edge_pairs = sorted({(min(u, v), max(u, v)) for u, v in edges.T.tolist() if u != v})
    device = solver.accelerator.device
    n = int(graph.num_nodes)
    graph = graph.to(device)
    adj = adjacency(graph)
    engine = PaddedSearch(solver, n_max=shape["nodes"], capacity=shape["capacity"], chunk=10,
                          device=device, decode_chunk=0, dense=True, capture=True)
    engine.load(graph, edges, 1017)
    checkpoints = []
    for step in range(0, 351, 10):
        if step:
            engine.step_block()
        mask = decode_top_best(solver.problem, engine.best_states, engine.best_energy,
                               adj, graph.b, n, 4)
        nodes = [i for i in range(n) if mask & (1 << i)]
        if any((mask >> u) & (mask >> v) & 1 for u, v in edge_pairs):
            raise RuntimeError(f"Infeasible decoded set at step {step}")
        checkpoints.append({"steps": step, "nodes": nodes})
    record = {"graph_id": "idx_0", "seed": 1017, "node_count": n, "edges": edge_pairs,
              "checkpoint_dir": str(args.save_dir), "device": torch.cuda.get_device_name(device),
              "dtype": str(solver.dtype), "particles": args.num_k, "decode_top": 4,
              "padded_nodes": shape["nodes"], "capacity": shape["capacity"],
              "checkpoints": checkpoints}
    args.case_file.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"Recorded {len(checkpoints)} feasible sets on {n} vertices", flush=True)


if __name__ == "__main__":
    main()
