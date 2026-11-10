"""Separate graph-wide add/drop offsets and node variation on matched B states."""

import argparse
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

import torch

from eval_v2 import VALID, dataset_at, write_run
from mis_v1 import adjacency
from v7_runs import preflight_policy


def top5_overlap(a, b):
    a = a.topk(5, dim=0).indices.T
    b = b.topk(5, dim=0).indices.T
    return (a[:, :, None] == b[:, None, :]).any(dim=-1).float().mean()


@torch.no_grad()
def diagnose(pilot, samples):
    baseline = preflight_policy("none", "no")
    device = baseline.accelerator.device
    methods = {arm: preflight_policy(f"v7_{arm}", "no", checkpoint=pilot / f"train_{arm}/checkpoints")
               for arm in ("mlp", "message")}
    data = dataset_at(VALID, baseline.dtype, samples)
    rows = {arm: [] for arm in methods}
    for i, graph in enumerate(data):
        graph = graph.to(device)
        adj = adjacency(graph)
        normalized = baseline.policy_topology(graph, adj)
        rng = torch.Generator(device=device).manual_seed(1017 + i)
        biases = {arm: policy.policy_topology(graph, adj).cached_action_bias for arm, policy in methods.items()}
        accum = {arm: [] for arm in methods}
        for step, state in enumerate(baseline.sample_states(graph, adj, generator=rng, normalized=normalized)):
            if step not in (25, 50, 75, 100):
                continue
            base = baseline.flip_logits(state, graph, adj, normalized)
            p = base.sigmoid()
            for arm, bias in biases.items():
                average = bias.mean(dim=0)
                full = base + torch.where(state.bool(), bias[:, 1:2], bias[:, 0:1])
                constant = base + torch.where(state.bool(), average[1], average[0])
                full_p = full.sigmoid()
                accum[arm].append(dict(
                    mean_abs_probability_shift=float((full_p-p).abs().mean()),
                    mean_abs_node_specific_probability_shift=float((full_p-constant.sigmoid()).abs().mean()),
                    delta_expected_flips=float((full_p-p).sum(dim=0).mean()),
                    top5_overlap=float(top5_overlap(base, full)),
                    top5_overlap_after_mean_control=float(top5_overlap(constant, full))))
        for arm, bias in biases.items():
            rows[arm].append(dict(
                graph_id=Path(data.files[i]).stem,
                mean_add_logit=float(bias[:, 0].mean()), mean_drop_logit=float(bias[:, 1].mean()),
                node_centered_energy_fraction=float((bias-bias.mean(dim=0)).square().sum() / bias.square().sum()),
                **{key: mean(item[key] for item in accum[arm]) for key in accum[arm][0]}))
    summary = {arm: {key: mean(row[key] for row in records) for key in records[0] if key != "graph_id"}
               for arm, records in rows.items()}
    output = pilot / "bias_diagnostics"
    output.mkdir(exist_ok=False)
    write_run(output, dict(samples=len(data), states=[25, 50, 75, 100], summary=summary, per_graph=rows))
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--samples", default=20, type=int)
    args = parser.parse_args()
    diagnose(args.pilot, args.samples)
