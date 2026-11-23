"""M6 步数匹配对照：固定步数（而不是固定时钟），比较 num_k。

部署 arm 在 1.0 s 里实际走 ~808 步；K 变小 → 每步更便宜 → 1 s 内步数更多。
于是「K 小更好」可能只是「步数更多」。这里两条臂都**恰好跑 S 步**
（S = 808 或部署臂实测步数），只留 num_k 一个变量，比较端点 SIZE。

与 eval_v8_padded.main 的差别只有三处，其余（padded/captured、dense、fp16、
warm start、A.3 解码、top-4 选取）逐项相同：
  1. ``engine.run(float("inf"), ..., max_blocks=ceil(S / chunk))`` —— 用步数而非时钟收尾；
  2. 不做末端 PAVE（PAVE 是固定后处理，两条臂一样；这里比较 search_size 更干净）；
  3. 每图多记 ``archive_best``（不解码，直接用 20/200 列档案的最大能量对应 size）。

用法：
  python train_scripts/v11/validation/k_sweep_steps.py --num_k 20 --fixed_steps 808 \
      --test_samples 200 --dense --decode_top 4 \
      --save_dir outputs/v2/none_seed0/checkpoints --result_dir outputs/v11_validation/steps808_k20
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))

from eval_v2 import dataset_at, load_policy, sync  # noqa: E402
from eval_v8_padded import build_parser, decode_top_best, shape_of  # noqa: E402
from mis_v1 import adjacency  # noqa: E402
from mis_v8_pad import PaddedSearch  # noqa: E402


def main():
    parser = build_parser()
    parser.add_argument("--fixed_steps", type=int, required=True,
                        help="两条臂都恰好跑这么多步（chunk 向上取整）")
    args = parser.parse_args()
    if not args.dense:
        raise SystemExit("M6 步数匹配对照必须在部署口径下跑：请加 --dense")
    args.arms = ["padded"]
    args.decoders = ["captured"]
    args.temperatures = [1.0]
    args.anneal = 0
    args.repeats = 1
    args.budgets = [0.0]

    solver = load_policy(args)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples,
                         official=args.split == "test")
    n_max, capacity = shape_of(dataset, args.padded_shapes)
    blocks_target = int(math.ceil(args.fixed_steps / args.chunk))
    print(f"padded shape: nodes {n_max}, capacity {capacity}; "
          f"num_k={args.num_k} fixed_steps={args.fixed_steps} "
          f"-> max_blocks={blocks_target} ({blocks_target * args.chunk} 步上限)", flush=True)

    engine = PaddedSearch(solver, n_max=n_max, capacity=capacity, chunk=args.chunk,
                          device=device, decode_chunk=args.decode_chunk, dense=args.dense)
    engine.load(dataset[0].to(device), dataset[0].edge_index, args.seeds[0])
    engine.run(perf_counter() + 0.05, lambda states: None)
    sync(device)

    extended = 20000
    solver.num_t = extended
    rows = []
    for index in range(len(dataset)):
        graph = dataset[index]
        graph_id = f"idx_{index}" if args.split == "test" else Path(dataset.files[index]).stem
        edges = graph.edge_index
        n = int(graph.num_nodes)
        graph = graph.to(device)
        adj = adjacency(graph)
        seed = args.seeds[0]
        sync(device)
        tick = perf_counter()
        incumbent = 0
        box = []

        def decode(best):
            nonlocal incumbent
            candidate = decode_top_best(solver.problem, best, engine.best_energy, adj,
                                        graph.b, n, args.decode_top)
            if candidate.bit_count() > incumbent:
                incumbent = candidate.bit_count()
                box.append(candidate)

        loaded = engine.load(graph, edges, seed)
        decode(engine.best_states)
        prelude = perf_counter() - tick
        steps, decode_seconds, blocks = engine.run(
            float("inf"), decode, max_blocks=blocks_target, decode_every=10 ** 9)
        sync(device)
        total = perf_counter() - tick
        best_bits = engine.decode_best(engine.best_states)
        rows.append(dict(
            graph_id=graph_id, num_k=args.num_k, fixed_steps=args.fixed_steps,
            steps=steps, blocks=blocks, nodes=n,
            search_size=incumbent, archive_best=best_bits.bit_count(),
            total_s=round(total, 4), prelude_s=round(prelude, 4),
            ms_per_step=round(1000.0 * (total - prelude) / max(steps, 1), 4),
        ))
        if (index + 1) % 25 == 0:
            mean_size = sum(r["search_size"] for r in rows) / len(rows)
            print(f"  [{index + 1}/{len(dataset)}] mean SIZE {mean_size:.4f} "
                  f"steps {steps} ms/step {rows[-1]['ms_per_step']:.3f}", flush=True)

    out = Path(args.result_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"steps{args.fixed_steps}_k{args.num_k}.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mean_size = sum(r["search_size"] for r in rows) / len(rows)
    mean_arch = sum(r["archive_best"] for r in rows) / len(rows)
    mean_ms = sum(r["ms_per_step"] for r in rows) / len(rows)
    print(f"done: {len(rows)} graphs, mean search_size {mean_size:.4f}, "
          f"archive_best {mean_arch:.4f}, steps {rows[-1]['steps']}, "
          f"ms/step {mean_ms:.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
