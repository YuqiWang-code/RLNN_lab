"""Deadline inference for the DIMES-MIS heatmap and masked categorical decoder."""
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
import torch


SAMPLE_BATCH = 32


def adjacency(n, edges):
    return coo_matrix((np.ones(edges.shape[1], np.float32), (edges[0], edges[1])),
                      shape=(n, n)).tocsr()


def independent_set(csr, order):
    free = np.ones(csr.shape[0], dtype=bool)
    selected = []
    for v in order:
        if free[v]:
            selected.append(v)
            free[v] = False
            free[csr.indices[csr.indptr[v]:csr.indptr[v+1]]] = False
    return np.sort(np.asarray(selected, dtype=np.int32))


def row_norm(x):
    centered = x - x.mean(dim=-1, keepdim=True)
    return centered * (centered.square().mean(dim=-1, keepdim=True) + 1e-20).rsqrt()


class DimesSearch:
    timing = "Host edge_index to copied host vertex sets. CSR, self loops, support, GPU inference and decoding are timed."

    def __init__(self, checkpoint, device="cuda:0"):
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        torch.set_float32_matmul_precision("highest")
        with np.load(checkpoint) as f:
            self.graph_weights = [(torch.from_numpy(f[f"w{i}_0"]).to(self.device),
                                   torch.from_numpy(f[f"w{i}_1"]).to(self.device)) for i in range(1, 21)]
            self.dense = {k.removeprefix("dense_"): torch.from_numpy(f[k]).to(self.device)
                          for k in f if k.startswith("dense_")}
            fixture = f["fixture_adjacency"]
            csr = adjacency(len(fixture), np.array(np.nonzero(fixture)))
            logits = self.predict(csr)
            reference = f["fixture_logits"].reshape(logits.shape)
            self.fixture_max_error = float(np.max(np.abs(logits - reference)))
            np.testing.assert_allclose(logits, reference, atol=5e-4, rtol=5e-4)
            expected_nodes = np.sort(f["fixture_greedy_nodes"].ravel())
            nodes = independent_set(csr, np.argsort(-reference[:, 0], kind="stable"))
            np.testing.assert_array_equal(nodes, expected_nodes)
            port_nodes = independent_set(csr, np.argsort(-logits[:, 0], kind="stable"))
            self.fixture_port_matches = bool(np.array_equal(port_nodes, expected_nodes))

    def prepare(self, graph):
        return graph.num_nodes, graph.edge_index.numpy()

    @torch.inference_mode()
    def predict(self, csr):
        n = csr.shape[0]
        support = csr.copy()
        support.setdiag(1)
        support.sort_indices()
        counts = np.diff(support.indptr)
        inv = 1.0 / np.sqrt(counts.astype(np.float32))
        support.data = -np.repeat(inv, counts) * inv[support.indices]
        support.setdiag(1 - inv * inv)
        lap = torch.sparse_csr_tensor(torch.from_numpy(support.indptr), torch.from_numpy(support.indices),
                                      torch.from_numpy(support.data), size=(n, n), device=self.device)
        h = torch.full((n, 32), 1/32, device=self.device)
        for w0, w1 in self.graph_weights[:-1]:
            h = (h @ w0 + torch.sparse.mm(lap, h @ w1)).relu()
        weights = self.dense
        h = row_norm(h @ weights["weights"])
        for i in range(1, 10):
            res = (h @ weights[f"weights_1_{i}"] + weights[f"bias_1_{i}"]).relu()
            h = row_norm(h + res @ weights[f"weights_2_{i}"])
        h = .02 * h
        for i in (10, 11):
            h = (h @ weights[f"weights_1_{i}"] + weights[f"bias_1_{i}"]).relu()
        w0, w1 = self.graph_weights[-1]
        out = h @ w0 + torch.sparse.mm(lap, h @ w1)
        pairs = out.reshape(n, -1, 2).log_softmax(dim=-1)
        logits = pairs[:, :, 1] - pairs[:, :, 0]
        return (logits / logits.std(dim=0, unbiased=False).clamp_min(1)).cpu().numpy()

    def warmup(self, raw):
        csr = adjacency(*raw)
        scores = self.predict(csr)
        independent_set(csr, np.argsort(-scores[:, 0], kind="stable"))

    def solve(self, raw, seed, tick, clock):
        deadline = tick + clock
        self.steps = 0
        csr = adjacency(*raw)
        logits = self.predict(csr)
        rng = np.random.default_rng(seed)
        best = -1
        for scores in logits.T:
            if perf_counter() >= deadline:
                return
            nodes = independent_set(csr, np.argsort(-scores, kind="stable"))
            self.steps += 1
            now = perf_counter()
            if now <= deadline and len(nodes) > best:
                best = len(nodes)
                yield now - tick, nodes
        batch_time = 0
        while perf_counter() + batch_time < deadline:
            start = perf_counter()
            for scores in logits.T:
                # Gumbel order samples the same masked categorical choices without replacement.
                perturbed = scores + rng.gumbel(size=(SAMPLE_BATCH, len(scores)))
                orders = np.argsort(-perturbed, axis=1)
                nodes = max((independent_set(csr, order) for order in orders), key=len)
                self.steps += SAMPLE_BATCH
                now = perf_counter()
                if now <= deadline and len(nodes) > best:
                    best = len(nodes)
                    yield now - tick, nodes
            batch_time = perf_counter() - start

    def protocol(self):
        return dict(checkpoint=str(self.checkpoint), revision="95aaa91c74cd47dc441c84eea9d38ead44f61ffd",
                    inference="GCN_DEEP_DIVER_DIMES FP32, self loops, normalize factor 1",
                    decoder="greedy followed by repeated best-of-batch masked categorical samples",
                    samples_per_batch=SAMPLE_BATCH, heads=self.graph_weights[-1][0].shape[1]//2,
                    steps="completed decoded sets", tf_fixture_max_abs_error=self.fixture_max_error,
                    fixture_decoder_matches_on_tf_logits=True, fixture_port_greedy_matches_upstream=self.fixture_port_matches,
                    test_time_meta_update=False)
