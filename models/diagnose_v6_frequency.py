"""Measure cross-layer energy and readout contributions on validation trajectories."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import torch

from eval_v2 import VALID, dataset_at, load_policy, policy_parser, write_run
from mis_v1 import adjacency


def dirichlet_ratio(h, adj):
    h = h.float().flatten(1)
    return float((h * (h - adj @ h)).sum() / h.square().sum().clamp_min(1e-12))


@torch.no_grad()
def diagnose(args):
    parser = policy_parser()
    mode = "none" if args.arm == "baseline" else args.arm
    options = parser.parse_args(["--result_dir", str(args.output), "--split", "validation",
        "--save_dir", str(args.checkpoint), "--frequency", mode, "--mixed_precision", "no"])
    solver = load_policy(options)
    model = solver.accelerator.unwrap_model(solver.model)
    device = solver.accelerator.device
    dataset = dataset_at(VALID, solver.dtype, args.samples)
    rows = []
    for index, graph in enumerate(dataset):
        gid = Path(dataset.files[index]).stem
        graph = graph.to(device)
        adj = adjacency(graph)
        topo = solver.policy_topology(graph, adj)
        rng = torch.Generator(device=device).manual_seed(1017 + index)
        states = solver.sample_states(graph, adj, generator=rng, normalized=topo)
        for step, state in enumerate(states):
            if step not in (0, 10, 50, 100):
                continue
            if mode == "none":
                h = model.initial_project(state.unsqueeze(-1))
                layers = []
                for conv, linear in zip(model.convs, model.linears):
                    h = h + model.activation(conv(h, topo) + linear(h))
                    layers.append(h)
                meso = high = None
                normalized = topo
            else:
                _, layers, meso, high = model(state, graph.edge_index, adj, topo, diagnose=True)
                normalized = topo.normalized
            row = dict(graph_id=gid, step=step,
                       meso_logit_energy=float(meso.square().mean()) if meso is not None else None,
                       high_logit_energy=float(high.square().mean()) if high is not None else None)
            row.update({f"layer{i}_dirichlet_ratio": dirichlet_ratio(h, normalized)
                        for i, h in enumerate(layers, 1)})
            rows.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per_state.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {key: mean(row[key] for row in rows) for key in rows[0]
               if key not in ("graph_id", "step") and rows[0][key] is not None}
    write_run(args.output, dict(status="completed", arm=args.arm, samples=len(dataset),
                               checkpoints=args.checkpoint, summary=summary))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    diagnose(parser.parse_args())
