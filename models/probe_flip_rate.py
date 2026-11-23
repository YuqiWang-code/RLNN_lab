"""How many nodes does the frozen sampler actually flip per step?

The gap to RLSA is in the dynamics, not in the budget: 2887 steps of our sampler
reach 19.983 while 300 annealed steps of RLSA already reach 19.962.  The cheapest
structural difference to check is the size of one move.  RLSA's kth-value gate
flips at most ``num_d`` nodes per column per step (5 of 264, ~2 %), while our
update flips each node independently with ``sigmoid(logit)`` and no cardinality
control.  If the measured flip fraction is far above a few percent, the sampler is
thrashing and a top-m gate is worth trying; if it is already small, the move-size
hypothesis is dead and the difference is somewhere else.
"""
import argparse
from pathlib import Path
from statistics import mean

import torch

import mis_v1
from eval_v2 import dataset_at, load_policy, policy_parser, sync

DEFAULT_POLICY = Path("/home/hzeng/project/ZH/RLNN_lab/outputs/v2/none_seed0/checkpoints")
DEFAULT_TEST = "models/data/MIS_tensors/RB_small_author_test"


def main():
    parser = policy_parser()
    parser.add_argument("--steps", type=int, default=200)
    parser.set_defaults(save_dir=str(DEFAULT_POLICY), test_path=DEFAULT_TEST,
                        result_dir=Path("outputs/v8_macro/flip_rate"))
    args = parser.parse_args()

    solver = load_policy(args)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples, official=True)
    seed = args.seed

    for index in range(len(dataset)):
        graph = dataset[index].to(device)
        adjacency = mis_v1.adjacency(graph)
        torus = torch.Generator(device=device).manual_seed(seed)
        normalized = solver.policy_topology(graph, adjacency)
        x = torch.randint(0, 2, (graph.num_nodes, solver.num_k), device=device,
                          dtype=solver.dtype, generator=torus)
        previous = x.clone()
        fractions, sizes = [], []
        for step in range(args.steps):
            product = adjacency @ x
            logits = solver.flip_logits(x, graph, adjacency, normalized)
            rr = torch.rand(x.shape, device=device, dtype=x.dtype, generator=torus)
            x = torch.where(rr < torch.sigmoid(logits), 1 - x, x)
            fractions.append(float((x != previous).float().mean()))
            previous = x.clone()
            if step % 50 == 49:
                sizes.append(float(x.mean(dim=0).mean()))
        sync(device)
        print(f"idx_{index}: nodes {graph.num_nodes} "
              f"flip_fraction mean {mean(fractions):.4f} "
              f"max {max(fractions):.4f} min {min(fractions):.4f} "
              f"(of {graph.num_nodes} nodes = {mean(fractions) * graph.num_nodes:.1f} per step); "
              f"density at 50/100/150/200: " + " ".join(f"{v:.3f}" for v in sizes), flush=True)


if __name__ == "__main__":
    main()
