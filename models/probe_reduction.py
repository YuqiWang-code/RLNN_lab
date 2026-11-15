"""How much does kernelization actually remove on RB graphs?

The plan is to rewrite the main loop KaMIS-style: fold the graph down with
reduction rules, solve the kernel, then lift the solution back.  That is worth
3-5 days only if the rules bite.  On random Barabasi-Albert graphs with average
degree 38 there are almost no degree-1 or degree-2 vertices, so this probe
measures the kernel size before anyone rewrites a loop.

Implemented rules (all exact for unweighted MIS):
  * isolated vertex      -> take it, delete it
  * pendant vertex       -> take it, delete it and its neighbour
  * domination           -> N[u] subset of N[v]  =>  delete v
  * closed twins         -> N[u] == N[v] (u,v adjacent, same closed nbrhd) => contract

    python models/probe_reduction.py --graphs 20 --split validation \
        --test_path models/data/MIS_tensors/RB_small_author_test
"""

import sys
from pathlib import Path
from statistics import mean

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_v2 import dataset_at  # noqa: E402


def bits(mask):
    while mask:
        bit = mask & -mask
        mask ^= bit
        yield bit.bit_length() - 1


def closed(neighbours, v):
    return neighbours[v] | (1 << v)


def reduce_once(neighbours, alive):
    """One sweep; returns (alive, taken, changed)."""
    taken = 0
    changed = False
    for v in list(bits(alive)):
        if not (alive >> v) & 1:
            continue
        degree = (neighbours[v] & alive).bit_count()
        if degree == 0:
            taken |= 1 << v
            alive &= ~(1 << v)
            changed = True
        elif degree == 1:
            u = (neighbours[v] & alive).bit_length() - 1
            taken |= 1 << v
            alive &= ~((1 << v) | (1 << u))
            changed = True
    for v in list(bits(alive)):
        if not (alive >> v) & 1:
            continue
        here = closed(neighbours, v) & alive
        for u in bits(neighbours[v] & alive):
            if closed(neighbours, u) & alive == here:
                alive &= ~(1 << v)  # closed twins: keep one of the pair
                changed = True
                break
    return alive, taken, changed

def kernel(neighbours, n):
    alive = (1 << n) - 1
    taken = 0
    for _ in range(50):
        alive, extra, changed = reduce_once(neighbours, alive)
        taken |= extra
        if not changed:
            break
    return alive, taken


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=20)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--test_path", default="models/data/MIS_tensors/RB_small_author_test")
    parser.add_argument("--reference_seed", type=int, default=0)
    args = parser.parse_args()

    dataset = dataset_at(args.test_path, torch.float32, args.graphs,
                         official=args.split == "test")
    rows = []
    for index in range(len(dataset)):
        graph = dataset[index]
        n = int(graph.num_nodes)
        edges = graph.edge_index
        neighbours = [0] * n
        for u, v in zip(edges[0].tolist(), edges[1].tolist()):
            neighbours[u] |= 1 << v
            neighbours[v] |= 1 << u
        alive, taken = kernel(neighbours, n)
        remaining = alive.bit_count()
        rows.append(dict(n=n, alive=remaining, taken=taken.bit_count(),
                         removed=n - remaining))
        print(f"{index + 1}/{len(dataset)} n {n} kernel {remaining} "
              f"({remaining / n:.1%}) taken {taken.bit_count()}", flush=True)

    print(f"\nmean: n {mean(r['n'] for r in rows):.1f} "
          f"kernel {mean(r['alive'] for r in rows):.1f} "
          f"({mean(r['alive'] / r['n'] for r in rows):.1%}) "
          f"taken {mean(r['taken'] for r in rows):.2f} "
          f"removed {mean(r['removed'] for r in rows):.2f}")


if __name__ == "__main__":
    main()
