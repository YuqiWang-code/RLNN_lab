"""Deadline evaluation of Learning What to Defer using the released actor and MIS environment."""
from time import perf_counter

import dgl
import torch
from env import MaximumIndependentSetEnv
from ppo.actor_critic import ActorCritic
from ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet


DEVICE = torch.device("cuda:0")
HORIZON = 32
MAX_NODES = 300
SAMPLES = 10


def actor_critic():
    return ActorCritic(PolicyGraphConvNet, ValueGraphConvNet, MAX_NODES, 128, 4, DEVICE)


class OfficialPPOLwD:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.policy = actor_critic().eval()
        self.policy.actor_net.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
        self.env = MaximumIndependentSetEnv(HORIZON, MAX_NODES, 0.0, DEVICE)

    def prepare(self, graph):
        result = dgl.graph(tuple(graph.edge_index), num_nodes=graph.num_nodes).to(DEVICE)
        result.create_formats_()
        torch.cuda.synchronize(DEVICE)
        return result

    def warmup(self, graph):
        list(self.solve(graph, 1017, perf_counter(), 1.0))

    @torch.no_grad()
    def solve(self, graph, seed, tick, clock):
        torch.manual_seed(seed)
        observation = self.env.register(graph, SAMPLES)
        self.steps = 0
        best = 0
        last_step = 0.0
        done = False
        while not done and perf_counter() + last_step < tick + clock:
            start = perf_counter()
            action = self.policy.act(observation, graph)
            observation, _, finished, info = self.env.step(action)
            largest = info["sol"][0].argmax()
            nodes = (self.env.x[:, largest] == 1).nonzero().flatten().cpu().numpy()
            done = bool(finished.all())
            now = perf_counter()
            self.steps += 1
            if len(nodes) > best and now <= tick + clock:
                best = len(nodes)
                yield now - tick, nodes
            last_step = now - start
        self.info = {"steps": self.steps, "all_trajectories_finished": done}

    def protocol(self):
        return dict(checkpoint=str(self.checkpoint), weights="RB-small retrained",
                    upstream_revision="174ce85e6ab9dda0d1579caa4ebdb98ff60edcc5",
                    device=str(DEVICE), dgl=dgl.__version__, samples=SAMPLES, max_steps=HORIZON,
                    delivery="largest nonempty accepted independent set after each synchronized action")
