"""Pure step cost of the padded engine as a function of the particle width.

``probe_k_scale`` showed that the *best-of-population* decode gains +0.25 from
K=20 to K=200 at a fixed 700 steps, so the deployment question is whether a wide
population still fits in the one-second clock.  ``eval_v8_padded`` decodes every
block by default, and a decode costs milliseconds at K=200, which hides the real
step cost; this probe runs the captured block with a no-op decoder and times it.
"""
from statistics import mean
from time import perf_counter

import torch

from eval_v8_padded import build_parser, shape_of
from eval_v2 import TEST, TEST_CACHE, dataset_at, load_policy
from mis_v8_pad import PaddedSearch


def main():
    parser = build_parser()
    parser.add_argument("--num_ks", type=int, nargs="+", default=[20, 50, 200, 800])
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 100, 700, 2000])
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.test_path is None:
        args.test_path = str(TEST_CACHE if (TEST_CACHE / "tensor_manifest.json").is_file() else TEST)

    solver = load_policy(args)
    dataset = dataset_at(args.test_path, solver.dtype, args.samples, official=True)
    nodes, capacity = shape_of(dataset, len(dataset))
    print(f"shapes n_max={nodes} capacity={capacity}", flush=True)

    for num_k in args.num_ks:
        solver.num_k = num_k
        engine = PaddedSearch(solver, n_max=nodes, capacity=capacity, chunk=10, decode_chunk=4)
        graph = dataset[0]
        engine.load(graph, graph.edge_index, args.seed)
        for steps in args.steps:
            blocks = max(1, steps // engine.chunk)
            engine.run(perf_counter() + 1e9, lambda states: None, max_blocks=blocks)
            torch.cuda.synchronize()
            started = perf_counter()
            engine.run(perf_counter() + 1e9, lambda states: None, max_blocks=blocks)
            torch.cuda.synchronize()
            seconds = perf_counter() - started
            per_step = 1e6 * seconds / (blocks * engine.chunk)
            print(f"K={num_k:<4} steps {blocks * engine.chunk:<5} {per_step:>8.2f} us/step "
                  f"{1000 * seconds / (blocks * engine.chunk):.3f} ms/step", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
