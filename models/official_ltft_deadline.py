"""Deadline adapter for the upstream GFlowNet-CombOpt MIS policy."""
from time import perf_counter

import dgl
from omegaconf import OmegaConf
import torch
from algorithm import DetailedBalanceTransitionBuffer
from util import get_mdp_class

from eval_v2 import sync


ROLLOUTS = 20


class OfficialLTFT:
    def __init__(self, checkpoint, config, device):
        self.device = torch.device(device)
        cfg = OmegaConf.load(config)
        cfg.task = "MaxIndependentSet"
        cfg.leaf_coef = cfg.lc
        self.policy = DetailedBalanceTransitionBuffer(cfg, self.device)
        self.policy.load(str(checkpoint))
        self.env = get_mdp_class(cfg.task)
        self.config = cfg
        self.checkpoint = checkpoint

    def prepare(self, graph):
        result = dgl.graph(tuple(graph.edge_index), num_nodes=graph.num_nodes).to(self.device)
        sync(self.device)
        return result

    def warmup(self, graph):
        list(self.solve(graph, 1017, perf_counter(), 0.2))

    @torch.no_grad()
    def solve(self, graph, seed, tick, clock):
        torch.manual_seed(seed)
        batch = dgl.batch([graph] * ROLLOUTS)
        env = self.env(batch, self.config)
        self.steps = 0
        last_step = 0.0
        best = 0
        finished = False
        while not finished and perf_counter() - tick + last_step < clock:
            start = perf_counter()
            action = self.policy.sample(batch, env.state, env.done, rand_prob=0.0)
            state = env.step(action).reshape(ROLLOUTS, graph.num_nodes())
            selected = state == 1
            largest = selected.sum(dim=1).argmax()
            nodes = selected[largest].nonzero().flatten().cpu().numpy()
            finished = bool(env.done.all())
            ready = perf_counter() - tick
            self.steps += 1
            last_step = ready - (start - tick)
            if ready <= clock and len(nodes) > best:
                best = len(nodes)
                yield ready, nodes

    def protocol(self):
        return dict(checkpoint=str(self.checkpoint), model="upstream forward-looking GFlowNet",
                    upstream_revision="feaa3b57f522e3a57a6a943bfa9bf1a2501eaf2d",
                    dgl=dgl.__version__, device=str(self.device), rollouts=ROLLOUTS,
                    delivery="largest legal complete or partial rollout after each synchronized action")
