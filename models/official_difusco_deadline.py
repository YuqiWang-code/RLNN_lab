"""Deadline evaluation of the released categorical DIFUSCO MIS checkpoint."""
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import scipy.sparse
import torch
from difusco.pl_meta_model import COMetaModel
from difusco.pl_mis_model import MISModel
from difusco.utils.diffusion_schedulers import InferenceSchedule
from difusco.utils.mis_utils import mis_decode_np


RELEASE_URL = "https://drive.google.com/file/d/1zsKEXOTjBGWn3dJxPSG-9QFPtbrrJ5J7/view"
REVISION = "65eb3b7bde76097c0d3438ce743fa6769baf9ce5"
SCHEDULES = (50, 20, 10, 5, 4, 3, 2, 1)


class InferenceModel(MISModel):
    def __init__(self, config):
        COMetaModel.__init__(self, config, node_feature_only=True)


class OfficialDIFUSCO:
    def __init__(self, graphs, checkpoint):
        self.checkpoint = checkpoint
        cfg = SimpleNamespace(diffusion_type="categorical", diffusion_schedule="linear",
                              diffusion_steps=1000, sparse_factor=-1, n_layers=12,
                              hidden_dim=256, aggregation="sum", use_activation_checkpoint=False)
        self.model = InferenceModel(cfg).cuda().eval()
        saved = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(saved["state_dict"])
        self.schedule = {}
        for steps in SCHEDULES:
            time_schedule = InferenceSchedule("cosine", T=self.model.diffusion.T, inference_T=steps)
            self.schedule[steps] = [tuple(np.asarray([t], dtype=int) for t in time_schedule(i))
                                    for i in range(steps)]
        self.calibration_graph = max(graphs, key=lambda graph: graph.edge_index.shape[1])
        self.inference_steps = None
        self.steps = 0

    def prepare(self, graph):
        n = graph.num_nodes
        vertices = torch.arange(n, dtype=torch.long)
        edges = torch.cat((graph.edge_index.long(), vertices.repeat(2, 1)), dim=1)
        adj = scipy.sparse.csr_matrix((np.ones(edges.shape[1], dtype=np.int8), edges.numpy()), shape=(n, n))
        return edges.cuda(), adj, n

    @torch.inference_mode()
    def decode(self, prepared, seed, steps):
        edges, adj, n = prepared
        torch.manual_seed(seed)
        state = (torch.randn(n, device=edges.device) > 0).long()
        for t, target_t in self.schedule[steps]:
            state = self.model.categorical_denoise_step(state, t, edges.device, edges, target_t)
        return np.flatnonzero(mis_decode_np(state.cpu().numpy() + 1e-6, adj))

    def warmup(self, prepared):
        prepared = self.prepare(self.calibration_graph)
        for steps in SCHEDULES:
            self.decode(prepared, 1017, steps)
            start = perf_counter()
            self.decode(prepared, 1017, steps)
            elapsed = perf_counter() - start
            print(f"DIFUSCO calibration: {steps} steps in {elapsed * 1000:.2f} ms", flush=True)
            self.inference_steps = steps
            if elapsed <= 0.18:
                break

    def solve(self, prepared, seed, tick, clock):
        self.steps = self.inference_steps
        nodes = self.decode(prepared, seed, self.steps)
        ready = perf_counter() - tick
        if ready <= clock:
            yield ready, nodes

    def protocol(self):
        return dict(checkpoint=str(self.checkpoint), checkpoint_source=RELEASE_URL,
                    checkpoint_dataset="SATLIB MIS", upstream_revision=REVISION, precision="fp32",
                    diffusion="categorical, 1000 training steps, cosine inference schedule",
                    inference_steps=self.inference_steps, calibration_candidates=SCHEDULES,
                    calibration_target_ms=180, sampling=1, decoding="official MIS greedy decoder")
