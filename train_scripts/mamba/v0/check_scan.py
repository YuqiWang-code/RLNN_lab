"""Trace native selective scan calls around CUDA Graph capture."""

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

import selective_scan_cuda
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from eval_v2 import TRAIN, dataset_at, search, write_run
from mis_v3_search import SearchGraph
from v7_runs import preflight_policy, provenance
from v7_structure import structure


@torch.no_grad()
def check(output):
    solver = preflight_policy("v7_mamba", "no", cuda_graph=True)
    graph = dataset_at(TRAIN, torch.float32, 1)[0]
    legal = SearchGraph(graph.num_nodes, graph.edge_index, seed=1017)
    _, groups = structure(graph.edge_index.numpy(), graph.num_nodes, solver.order_seed + graph.idx, order="bfs")
    graph = graph.to("cuda")
    search(solver, graph, legal, 1017)
    native = selective_scan_cuda.fwd
    calls = []

    def counted(*args, **kwargs):
        calls.append(torch.cuda.is_current_stream_capturing())
        with record_function("v7_native_selective_scan"):
            return native(*args, **kwargs)

    with patch.object(selective_scan_cuda, "fwd", counted), profile(activities=[ProfilerActivity.CPU]) as trace:
        solution, steps = search(solver, graph, legal, 1017)
        torch.cuda.synchronize()
    events = sum(event.name == "v7_native_selective_scan" for event in trace.events())
    assert len(calls) == len(groups) == events and not any(calls)
    assert steps == 100 and legal.feasible(solution)
    output.mkdir(parents=True, exist_ok=False)
    write_run(output, dict(**provenance(0), status="completed", steps=steps, native_scan_calls=events,
                           sequence_batches=[list(group.shape) for group in groups],
                           native_calls_during_capture=sum(calls), scan_calls_in_replay=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    check(parser.parse_args().output)
