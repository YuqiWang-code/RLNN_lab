"""Train the upstream PPO actor on independent RB graphs and select it on held-out RB graphs."""
import argparse
from pathlib import Path
import pickle
import random

import dgl
import numpy as np
import torch
from env import MaximumIndependentSetEnv
from ppo.framework import ProxPolicyOptimFramework
from ppo.storage import RolloutStorage
from official_ppo_lwd_deadline import actor_critic, DEVICE, HORIZON, MAX_NODES, SAMPLES


def read_graph(path):
    with path.open("rb") as stream:
        graph = pickle.load(stream)
    return dgl.from_networkx(graph)


@torch.no_grad()
def validate(policy, validation):
    scores = []
    env = MaximumIndependentSetEnv(HORIZON, MAX_NODES, 0.0, DEVICE)
    policy.eval()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(813)
        for graph in validation:
            ob = env.register(graph, SAMPLES)
            while True:
                ob, _, done, info = env.step(policy.act(ob, graph))
                if bool(done.all()):
                    scores.extend(info["sol"].max(dim=1).values.tolist())
                    break
    policy.train()
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    files = sorted(args.train.glob("*.gpickle"))
    valid = sorted(args.validation.glob("*.gpickle"))[:64]
    if len(files) != 4000 or len(valid) != 64:
        raise ValueError("Expected 4,000 independent training graphs and a held-out validation split")
    validation = [dgl.batch([read_graph(path) for path in valid[i:i+8]]).to(DEVICE)
                  for i in range(0, len(valid), 8)]
    policy = actor_critic()
    env = MaximumIndependentSetEnv(HORIZON, MAX_NODES, 0.1, DEVICE)
    framework = ProxPolicyOptimFramework(policy, 1e-4, 0.2, 4, 16, 0.5, 0.1, 0.5, DEVICE)
    batch_size = 32
    update = 0
    best = -1.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        random.shuffle(files)
        for i in range(0, len(files), batch_size):
            graph = dgl.batch([read_graph(path) for path in files[i:i+batch_size]]).to(DEVICE)
            rollout = RolloutStorage(HORIZON, graph.batch_size, 2)
            ob = env.register(graph, 2)
            rollout.insert_ob_and_g(ob, graph)
            with torch.no_grad():
                for _ in range(HORIZON):
                    actions, logp, values = policy.act_and_crit(ob, graph)
                    ob, reward, done, info = env.step(actions)
                    rollout.insert_tensors(ob, actions, logp, values, reward, done)
                    if bool(done.all()):
                        break
            rollout.compute_rets_and_advantages(1.0)
            losses = framework.update(rollout)
            update += 1
            print("update", update, "mean_train_size", info["sol"].max(dim=1).values.mean().item(),
                  "losses", [loss.item() for loss in losses], flush=True)
            if update % 25 == 0:
                score = validate(policy, validation)
                if score > best:
                    best = score
                    torch.save(policy.actor_net.state_dict(), args.checkpoint)
                print("validation", update, "size", score, "best", best, flush=True)


if __name__ == "__main__":
    main()
