"""Cross-graph probes for static information using frozen B trajectories."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F

from eval_v2 import TRAIN, dataset_at, search, write_run
from main import set_seed
from mis_v3_search import SearchGraph, nodes
from mis_v7_context import V7ActionPrior
from utils import build_batch
from v7_runs import preflight_policy, provenance
from v7_structure import FEATURES, structure, v7_topology


ORDER_SEED = 7301


def normalizer(train):
    values = []
    for graph in train:
        x, _ = structure(graph.edge_index.numpy(), graph.num_nodes, ORDER_SEED + graph.idx)
        values.append(x)
    values = np.concatenate(values)
    return values.mean(0), values.std(0).clip(min=1e-4)


def elite_frequency(masks, n):
    elite = sorted(masks, key=int.bit_count, reverse=True)[:(len(masks)+1)//2]
    y = np.zeros(n, dtype=np.float32)
    for mask in elite:
        y[list(nodes(mask))] += 1 / len(elite)
    return y


@torch.no_grad()
def collect(policy, dataset, indices, starts, random):
    samples = []
    for index in indices:
        cpu = dataset[index]
        edge_index = cpu.edge_index
        gid = Path(dataset.files[index]).stem
        features, _ = structure(edge_index.numpy(), cpu.num_nodes, ORDER_SEED + cpu.idx)
        graph = cpu.clone().to(policy.accelerator.device)
        masks, inputs = [], []
        take = torch.tensor(random.choice(cpu.num_nodes, size=min(32, cpu.num_nodes), replace=False),
                            device=graph.b.device)
        column = torch.tensor(random.integers(0, policy.num_k, size=len(take)), device=graph.b.device)

        def observe(step, state, _best, adj, normalized):
            if step not in (25, 50, 75, 100):
                return
            selected = 2 * (adj @ state)
            counts = selected[take, column]
            x = state[take, column]
            degree = adj.crow_indices().diff()[take]
            gain = (1 - 2 * x) * (policy.problem.beta * counts - 1)
            logits = policy.flip_logits(state, graph, adj, normalized)[take, column]
            inputs.append(torch.stack((x, counts, degree, counts/degree.clamp_min(1), gain, logits), 1).cpu().numpy())

        for rep in range(starts):
            seed = 40001 + 1009 * int(index) + rep
            legal = SearchGraph(cpu.num_nodes, edge_index, seed=seed)
            incumbent, _ = search(policy, graph, legal, seed, observe=observe if rep == 0 else None)
            assert legal.feasible(incumbent)
            masks.append(incumbent)
        a, b = elite_frequency(masks[::2], cpu.num_nodes), elite_frequency(masks[1::2], cpu.num_nodes)
        reliability = float(spearmanr(a, b).statistic)
        samples.append(dict(graph=cpu, graph_id=gid, proxy=elite_frequency(masks, cpu.num_nodes),
                            b1_inputs=np.concatenate(inputs), b1_targets=np.tile(features[take.cpu().numpy()], (4, 1)),
                            reliability=reliability if np.isfinite(reliability) else 0.0))
        if len(samples) % 20 == 0:
            print(f"Proxy labels {len(samples)}/{len(indices)}", flush=True)
    return samples


def b1(training, heldout):
    def values(split, key):
        return np.concatenate([item[key] for item in split])
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(values(training, "b1_inputs"), values(training, "b1_targets"))
    score = r2_score(values(heldout, "b1_targets"), ridge.predict(values(heldout, "b1_inputs")),
                     multioutput="raw_values")
    return dict(zip(FEATURES, score.tolist()))


def proxy_metrics(target, pred):
    labels = np.concatenate((np.ones_like(target), np.zeros_like(target)))
    auc = roc_auc_score(labels, np.tile(pred, 2), sample_weight=np.concatenate((target, 1-target)))
    correlation = spearmanr(target, pred).statistic
    return dict(auc=float(auc), spearman=float(correlation) if np.isfinite(correlation) else 0.,
                brier=float(np.mean((target-pred)**2)))


def fit(mode, local, training, heldout, mean, std, args, device):
    set_seed(7302)
    adapter = V7ActionPrior(mode, args.width, args.d_state).to(device)
    adapter.feature_mean.copy_(torch.as_tensor(mean, device=device))
    adapter.feature_std.copy_(torch.as_tensor(std, device=device))

    def prepare(samples):
        graph = build_batch([item["graph"] for item in samples]).to(device)
        topo = v7_topology(graph, adapter.normalize, ORDER_SEED, order=adapter.order)
        if local:
            topo = topo._replace(features=topo.features.clone())
            topo.features[:, 3:] = 0
        target = torch.as_tensor(np.concatenate([item["proxy"] for item in samples]), device=device)
        return topo, target

    batches = [prepare(training[i:i+8]) for i in range(0, len(training), 8)]
    testing = [prepare([item]) for item in heldout]
    optimizer = torch.optim.Adam(adapter.parameters(), lr=.003)
    for _ in range(args.proxy_epochs):
        for topo, target in batches:
            pred = adapter.out_proj(adapter.embeddings(topo))[:, 0]
            loss = F.binary_cross_entropy_with_logits(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return [dict(graph_id=item["graph_id"], **proxy_metrics(item["proxy"],
                    adapter.out_proj(adapter.embeddings(topo))[:, 0].sigmoid().cpu().numpy()))
                for item, (topo, _) in zip(heldout, testing)]


def run(args):
    solver = preflight_policy("none", "no", cuda_graph=True)
    dataset = dataset_at(TRAIN, torch.float32, 1000)
    random = np.random.default_rng(7303)
    samples = collect(solver, dataset, random.permutation(len(dataset))[:args.samples], args.starts, random)
    cut = int(len(samples)*.8)
    training, heldout = samples[:cut], samples[cut:]
    # P0 fit and normalization exclude the cross-graph proxy holdout.
    feature_mean, feature_std = normalizer(item["graph"] for item in training)
    arms = (("local_mlp", "v7_mlp", True), ("full_mlp", "v7_mlp", False),
            ("message", "v7_message", False), ("mamba", "v7_mamba", False))
    results = {}
    for label, mode, local in arms:
        results[label] = fit(mode, local, training, heldout, feature_mean, feature_std, args, solver.accelerator.device)
        print(label, np.mean([row["auc"] for row in results[label]]), flush=True)
    comparisons = {}
    indices = random.integers(0, len(heldout), (4000, len(heldout)))
    for control in ("local_mlp", "full_mlp", "message"):
        diff = np.asarray([row["auc"]-base["auc"] for row, base in zip(results["mamba"], results[control])])
        ci = np.quantile(diff[indices].mean(1), [.025, .975])
        comparisons[control] = dict(delta_auc=float(diff.mean()), ci95=ci.tolist())
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per_graph.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "graph_id", "auc", "spearman", "brier"])
        writer.writeheader()
        writer.writerows(dict(arm=arm, **row) for arm, rows in results.items() for row in rows)
    reliability = float(np.mean([item["reliability"] for item in heldout]))
    gate = all(comparisons[name]["delta_auc"] > 0 for name in ("local_mlp", "full_mlp", "message"))
    write_run(args.output, dict(**provenance(0), status="completed", starts=args.starts,
                               proxy_epochs=args.proxy_epochs, graph_train=[item["graph_id"] for item in training],
                               graph_heldout=[item["graph_id"] for item in heldout],
                               b1_heldout_r2=b1(training, heldout), mean_label_split_spearman=reliability,
                               mean_metrics={arm: {key: float(np.mean([row[key] for row in rows]))
                                                  for key in ("auc", "spearman", "brier")}
                                             for arm, rows in results.items()},
                               paired_delta_auc=comparisons, information_direction=gate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--proxy-epochs", type=int, default=25)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
