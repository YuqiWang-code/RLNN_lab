"""Verify frozen weights and the same-checkpoint node-term intervention."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "models"))

import torch

from eval_v2 import VALID, dataset_at
from mis_static_prior import cheap_features
from mis_v1 import adjacency
from utils import build_batch
from v7_runs import backbone, pave, digest, preflight_policy
from v7_structure import structure


@torch.no_grad()
def check(root, ref):
    original = torch.load(backbone(ref), map_location="cpu", weights_only=True)
    manifests, cohorts = [], []
    for arm in ("constants", "centered"):
        run = root / f"train_{arm}_ref{ref}"
        data = json.loads((run / "run.json").read_text(encoding="utf-8"))
        state = torch.load(run / "checkpoints/best.pt", map_location="cpu", weights_only=True)
        assert data["status"] == "completed" and data["updates"] == 512
        assert data["pave_sha256"] == digest(pave(ref))
        assert all(torch.equal(state[key], value) for key, value in original.items())
        manifests.append(data["teacher_state_sha256"])
        cohorts.append(data["validation_graph_ids"])
        assert data["first_gradients"][0]["constants_grad_norm"] > 0
        if arm == "centered":
            assert data["first_gradients"][1]["encoder_grad_norm"] > 0
    assert manifests[0] == manifests[1] and len(manifests[0]) == 20
    dataset = dataset_at(VALID, torch.float32)
    assert cohorts[0] == cohorts[1] == [Path(file).stem for file in dataset.files[50:100]]
    solver = preflight_policy("static_centered", "no", ref=ref,
                              checkpoint=root / f"train_centered_ref{ref}/checkpoints")
    device = solver.accelerator.device
    adapter = solver.accelerator.unwrap_model(solver.model).adapter
    graphs = [dataset[50], dataset[51]]
    for graph in graphs:
        features, _ = structure(graph.edge_index.numpy(), graph.num_nodes, 0)
        raw = cheap_features(adjacency(graph), (0, graph.num_nodes))
        torch.testing.assert_close(raw, torch.from_numpy(features[:, [0, 1, 2, 6, 7]]), atol=2e-3, rtol=1e-4)
    graphs = [graph.to(device) for graph in graphs]
    singles = [solver.policy_topology(graph, adjacency(graph)).cached_action_bias for graph in graphs]
    batch = build_batch(graphs)
    both = solver.policy_topology(batch, adjacency(batch)).cached_action_bias
    torch.testing.assert_close(both, torch.cat(singles), atol=2e-5, rtol=1e-5)
    adapter.centered = False
    masked = solver.policy_topology(batch, adjacency(batch))
    assert masked.features is None and masked.cached_action_bias.shape == (1, 2)
    for bias in singles:
        torch.testing.assert_close(bias.mean(dim=0, keepdim=True), masked.cached_action_bias, atol=1e-7, rtol=1e-5)
        assert (bias.abs() <= .5).all()
    print("Frozen weights, matched teachers, feature parity, graph centering and masked action bias verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ref", type=int, default=0)
    args = parser.parse_args()
    check(args.root, args.ref)
