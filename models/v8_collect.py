"""Collect frozen-teacher macro blocks for v8 ACMD.

Runs the frozen RLNN once per graph, cuts the trajectory into ``h``-step blocks at
a fixed set of start steps, and stores ``(x, Y, B)`` with their energies.  Labels
are produced only by the frozen policy, so the stored tensors are the teacher's
own behaviour, not an oracle.

    python -u models/v8_collect.py --reference_seed 0 --save_dir outputs/v2/none_seed0/checkpoints \
        --result_dir outputs/v8_macro --split validation --test_samples 100 --horizon 2 --seeds 1017
"""
import json
from pathlib import Path

import torch

from eval_v2 import dataset_at, load_policy, sync, write_run
from main import set_seed
from mis_v1 import adjacency
from mis_v8_macro import teacher_blocks
from v7_runs import digest


def build_parser():
    import eval_v3

    parser = eval_v3.build_parser()          # horizon, result_dir, save_dir, split, label
    for action in parser._actions:
        if action.dest == "label":
            action.required = False
    parser.set_defaults(split="validation", label="v8_collect", mixed_precision="no",
                        post=["none"], num_t=100)
    parser.add_argument("--data_split", choices=("train", "validation"), default="train")
    parser.add_argument("--reference_seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1017])
    parser.add_argument("--starts", type=int, nargs="+", default=[10, 25, 40, 55, 70, 85])
    return parser


def main():
    args = build_parser().parse_args()
    args.test_path = None
    args.seed = args.seeds[0]
    set_seed(args.seed)
    if args.data_split == "train":
        from eval_v2 import TRAIN
        args.test_path = str(TRAIN)        # load_policy only knows validation/test paths
    solver = load_policy(args)
    solver.model.requires_grad_(False)
    solver.num_t = args.num_t
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples)
    out = args.result_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, graph in enumerate(dataset):
        graph_id = Path(dataset.files[index]).stem
        tensor = graph.to(device)
        adj = adjacency(tensor)
        generator = torch.Generator(device=device).manual_seed(args.seeds[0] + index)
        blocks = teacher_blocks(solver, tensor, adj, generator, horizon=args.horizon,
                                starts=args.starts, num_t=args.num_t)
        sync(device)
        payload = []
        for block in blocks:
            payload.append(dict(state=block.state.to(torch.uint8).cpu(),
                                continuation=block.continuation.to(torch.uint8).cpu(),
                                record=block.record.to(torch.uint8).cpu(),
                                record_index=block.record_index.cpu(),
                                energy_start=block.energy_start.float().cpu(),
                                energy_end=block.energy_end.float().cpu(),
                                energy_record=block.energy_record.float().cpu()))
        rows.append(dict(graph_id=graph_id, blocks=payload, count=len(payload),
                         record_gap=float(sum((b["energy_record"] - b["energy_end"]).sum().item()
                                              for b in payload) / max(len(payload), 1)),
                         edge_index=tensor.edge_index.cpu()))
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(dataset)} graphs, {sum(r['count'] for r in rows)} blocks", flush=True)
    torch.save(rows, out / f"blocks_{args.data_split}_ref{args.reference_seed}.pt")
    write_run(args.result_dir, dict(version="v8_macro", status="completed", mode="collect",
                        ref=args.reference_seed, split=args.data_split, graphs=len(rows),
                        horizon=args.horizon, starts=args.starts, seeds=args.seeds,
                        blocks=sum(row["count"] for row in rows),
                        mean_record_gap=sum(row["record_gap"] for row in rows) / max(len(rows), 1),
                        checkpoint_sha256=digest(Path(args.save_dir) / "best.pt"),
                        config={k: str(v) for k, v in vars(args).items()}))
    print(json.dumps(dict(graphs=len(rows), blocks=sum(row["count"] for row in rows))), flush=True)


if __name__ == "__main__":
    main()
