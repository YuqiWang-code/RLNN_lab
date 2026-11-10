"""Check shared rollout labels, chunking, candidate strata and P3 output prefixes."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))
from models import GCN
from archived_frequency.mis_v4_frequency import V4FrequencyGCN, state_bands, v4_topology
from problems.mis import Problem
from archived_frequency.probe_v4_utility import paired_utilities as original_paired_utilities
from archived_frequency.probe_v4_utility_followup import (
    LABEL_MODES, build_parser, continuation_noise, graph_average, label_reliability,
    node_masks, paired_utilities, prefix_counts, probe, sample_candidates,
)


def run(device):
    torch.manual_seed(17)
    torch.set_num_threads(1)
    pairs = [(0, 1), (0, 2), (1, 2), (3, 4)]
    edges = torch.tensor(pairs + [(b, a) for a, b in pairs], device=device).T.contiguous()
    adj = torch.sparse_coo_tensor(edges, torch.full((edges.shape[1],), .5, device=device), (7, 7)).to_sparse_csr()
    topology = v4_topology(edges, adj, 7)
    graph = SimpleNamespace(num_nodes=7, edge_index=edges, b=-torch.ones(7, 1, device=device))
    state = torch.tensor([1, 1, 0, 1, 0, 1, 0], device=device).float()
    candidates = torch.tensor([0, 2, 4, 6], device=device)
    teacher = GCN(1, 128, 5, 1).to(device).eval().requires_grad_(False)
    problem = Problem(1.02)
    args = (teacher, state, candidates, graph, adj, topology.normalized, problem, 10, 12345)
    expected, expected_gains = original_paired_utilities(*args, replicates=2)
    for chunk in (1, 7, 256):
        sizes, gains = paired_utilities(*args, replicates=2, chunk_size=chunk)
        torch.testing.assert_close(sizes, expected, rtol=0, atol=0)
        torch.testing.assert_close(gains, expected_gains, rtol=0, atol=0)
    full_sizes, full_gains = paired_utilities(*args, replicates=16, chunk_size=7)
    for repeats in prefix_counts(16):
        sizes, gains = paired_utilities(*args, replicates=repeats, chunk_size=7)
        torch.testing.assert_close(sizes, full_sizes[:repeats], rtol=0, atol=0)
        torch.testing.assert_close(gains, full_gains[:repeats], rtol=0, atol=0)
    torch.testing.assert_close(continuation_noise(7, 10, 2, 12345, device),
                               continuation_noise(7, 10, 16, 12345, device)[:2], rtol=0, atol=0)
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    permuted, _ = paired_utilities(teacher, state, candidates[permutation], graph, adj,
        topology.normalized, problem, 10, 12345, replicates=16, chunk_size=7)
    torch.testing.assert_close(permuted[:, 0], full_sizes[:, 0], rtol=0, atol=0)
    torch.testing.assert_close(permuted[:, 1:], full_sizes[:, 1:][:, permutation], rtol=0, atol=0)

    details = state_bands(state[:, None], topology)
    masks, threshold = node_masks(state, details["c"][:, 0], details["r"][:, 0])
    assert torch.stack([masks[key] for key in list(masks)[:5]]).sum(0).eq(1).all()
    assert all(masks[key].any() for key in list(masks)[:5])
    assert threshold == 1 and masks["release_high"].nonzero().flatten().tolist() == [3]
    for sampling in ("uniform", "stratified"):
        selected = sample_candidates(masks, 6, sampling, 112)
        assert len(selected) == 6 and len(selected.unique()) == 6
        assert torch.equal(selected, selected.sort().values)
        assert torch.equal(selected, sample_candidates(masks, 6, sampling, 112))
        assert sample_candidates(masks, 24, sampling, 112).tolist() == list(range(7))
    baseline_energy, _ = problem.energy_func(adj, graph.b, state[:, None])
    for node in range(7):
        changed = state.clone()
        changed[node] = 1 - changed[node]
        energy, _ = problem.energy_func(adj, graph.b, changed[:, None])
        torch.testing.assert_close(energy - baseline_energy, details["delta_energy"][node], rtol=1e-5, atol=1e-6)
    stable = label_reliability(torch.tensor([[1., 0., -1.]]).repeat(8, 1))
    assert stable["half_pair_direction_agreement"] == 1 and stable["half_joint_non_tied_pairs"] == 3
    assert stable["mean_gain_sem"] == 0 and stable["half_top1_same"] == 1
    tied = label_reliability(torch.zeros(8, 4))
    assert tied["informative"] == 0 and tied["half_spearman"] is None
    assert graph_average([dict(graph_id="a", v=0.), dict(graph_id="a", v=1.), dict(graph_id="b", v=1.)], "v") == .75

    with tempfile.TemporaryDirectory(prefix="rlnn_p3_") as directory:
        temporary = Path(directory)
        data = temporary / "data"
        data.mkdir()
        torch.save(dict(edge_index=edges.cpu(), num_nodes=7), data / "toy.pt")
        (data / "tensor_manifest.json").write_text(json.dumps(dict(graphs=[dict(tensor="toy.pt", source="toy.gpickle")])))
        cli = ["--device", str(device), "--graphs", "1", "--nodes", "7", "--replicates", "8",
               "--phases", "0", "--horizon", "10", "--sampling", "stratified", "--chunk-size", "7",
               "--data-path", str(data), "--result-dir", str(temporary / "result")]
        for name, mode in LABEL_MODES.items():
            model = GCN(1, 128, 5, 1) if mode == "none" else V4FrequencyGCN(128, 5, mode=mode)
            path = temporary / f"{name}.pt"
            torch.save(model.state_dict(), path)
            cli.extend(["--model", name, mode, str(path)])
        metadata = probe(build_parser().parse_args(cli))
        assert metadata["status"] == "completed" and metadata["states"] == 1
        assert metadata["replicate_prefixes"] == [2, 4, 8]
        assert metadata["actions"] == 7 and metadata["action_replicates"] == 56
        result = temporary / "result"
        repeat_rows = list(csv.DictReader((result / "action_replicate.csv").open()))
        aggregate_rows = list(csv.DictReader((result / "action_aggregate.csv").open()))
        for row in aggregate_rows:
            samples = torch.tensor([float(x["gain"]) for x in repeat_rows
                                    if x["node"] == row["node"] and int(x["replicate"]) < int(row["R"])])
            assert float(row["gain_mean"]) == float(samples.mean())
            assert float(row["gain_std"]) == float(samples.std(unbiased=True))
            assert abs(float(row["gain_sem"]) - float(samples.std(unbiased=True)) / len(samples)**.5) < 1e-12
        ranked = list(csv.DictReader((result / "ranking.csv").open()))
        assert {row["model"] for row in ranked} == set(LABEL_MODES) | {
            "F4_no_low", "F4_no_high", "Energy", "Random", "DegreeLow", "SelectedNeighbors", "Release"}
    return dict(status="passed", device=str(device), checks=[
        "R2 original paired-label semantics", "chunk sizes 1/7/256", "R2/4/8/16 shared prefixes",
        "candidate permutation", "unique stratified sampling", "exact single-flip energy",
        "split-half reliability", "graph-weighted summaries", "CLI CSV prefixes and repeat moments",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(torch.device(args.device))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
