"""Deadline adapter for the released INTEL GCN and randomized prefix tree search."""
from time import perf_counter

import numpy as np
import torch


class IntelTreeSearch:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.device = torch.device("cuda:0")
        torch.set_float32_matmul_precision("highest")
        with np.load(checkpoint) as weights:
            self.layers = [tuple(torch.from_numpy(weights[f"w{layer}_{support}"]).to(self.device)
                                 for support in range(2)) for layer in range(1, 21)]
            fixture = torch.from_numpy(weights["fixture_adjacency"]).to(self.device)
            prediction = self.predict(fixture)
            self.tf_max_error = float(np.abs(prediction - weights["fixture_maps"]).max())
            np.testing.assert_allclose(prediction, weights["fixture_maps"], rtol=5e-4, atol=2e-4)

    def prepare(self, graph):
        adjacency = np.zeros((graph.num_nodes, graph.num_nodes), dtype=bool)
        source, target = graph.edge_index.numpy()
        adjacency[source, target] = True
        gpu = torch.from_numpy(adjacency).to(self.device, dtype=torch.float32)
        torch.cuda.synchronize(self.device)
        return adjacency, gpu

    @torch.no_grad()
    def predict(self, adjacency):
        n = adjacency.shape[0]
        degree = adjacency.sum(1).clamp_min(1).rsqrt()
        laplacian = torch.eye(n, device=self.device) - degree[:, None] * adjacency * degree[None, :]
        features = adjacency.new_full((n, 32), 1 / 32)
        for layer, (weight_self, weight_lap) in enumerate(self.layers):
            features = features @ weight_self + laplacian @ (features @ weight_lap)
            if layer < len(self.layers) - 1:
                features.relu_()
        return features.view(n, 32, 2).softmax(dim=2)[:, :, 1].cpu().numpy()

    def warmup(self, graph):
        list(self.solve(graph, 1017, perf_counter(), 0.2))

    def solve(self, prepared, seed, tick, clock):
        adjacency, gpu_adjacency = prepared
        rng = np.random.RandomState(seed)
        deadline = tick + clock
        root = ((), np.ones(len(adjacency), dtype=bool))
        queue = [root]
        best = 0
        self.steps = 0
        complete = 0
        inference_time = 0.0
        while perf_counter() + inference_time < deadline:
            start = perf_counter()
            selected, remaining = queue.pop(rng.randint(len(queue))) if queue else root
            ids = np.flatnonzero(remaining)
            idx = torch.from_numpy(ids).to(self.device)
            maps = self.predict(gpu_adjacency.index_select(0, idx).index_select(1, idx))
            inference_time = perf_counter() - start
            self.steps += 1
            for scores in maps.T:
                if perf_counter() >= deadline:
                    break
                nodes = list(selected)
                available = remaining.copy()
                for node in ids[np.argsort(1 - scores)]:
                    if not available[node]:
                        break
                    nodes.append(node)
                    available[node] = False
                    available[adjacency[node]] = False
                    if rng.random_sample() > 0.7 and available.any():
                        queue.append((tuple(nodes), available.copy()))
                if len(nodes) > best:
                    candidate = np.asarray(nodes, dtype=np.int64)
                    ready = perf_counter()
                    if ready <= deadline:
                        best = len(candidate)
                        yield ready - tick, candidate
                if available.any():
                    queue.append((tuple(nodes), available))
                else:
                    complete += 1
        self.info = {"gcn_expansions": self.steps, "complete_branches": complete}

    def protocol(self):
        return dict(checkpoint=str(self.checkpoint), weights="released INTEL model/model.ckpt",
                    upstream_revision="994789899fd05dcc3e3bd3d8a1bc73d7a6994691",
                    inference="FP32 dense PyTorch port, compared against released TensorFlow meta graph",
                    tf_fixture_max_absolute_error=self.tf_max_error, device=str(self.device),
                    maps=32, prefix_branch_probability=0.3, queue="uniform random", workers=1,
                    reduction=False, local_search=False, delivery="largest legal completed map prefix on host")
