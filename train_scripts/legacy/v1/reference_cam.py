"""Run the author's CAM and PyTorch DiffUCO solvers on the shared RB split."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "reference_methods/CAM/source"
sys.path[:0] = [str(SOURCE), str(ROOT / "models")]
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import torch

from eval_v1 import TEST, add_evaluation_arguments, evaluate

sys.path.remove(str(ROOT / "models"))
from solvers import CAM, DiffUCO


def main():
    parser = add_evaluation_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--method", choices=["CAM", "DiffUCO"], default="CAM")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mini_batch_size", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--valid_samples", type=int, default=100)
    parser.add_argument("--test_samples", type=int, default=-1)
    parser.add_argument("--num_t", type=int, default=50)
    args = parser.parse_args()
    args.result_dir = args.result_dir.resolve()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    settings = dict(
        train_path=str(ROOT / "models/data/MIS/RB_small_train"),
        valid_path=str(ROOT / "models/data/MIS/RB_small_val"),
        test_path=str(TEST), save_dir=str(args.result_dir / "checkpoints"),
        problem="mis", model="EncodeProcessDecode", num_workers=4, data_precision="fp32",
        num_l=8, num_h=64, num_k=20, num_kp=10, num_tp=50, time_scale=1000,
        repetition=1, skip_decode=False, beta=1.02, mis_decoding="continuous",
        num_d=5, num_clips=1, lambd=0, tau0=0.1, tau1=0, temp_schedule="linear",
        N_warmup=0, N_anneal=args.epochs, N_equil=0, patience=args.epochs,
        lr=0.0001, optimizer="AdamW", lr_scheduler="cosine-decay", weight_decay=0.0001,
        grad_clip=None, gradient_accumulation_steps=1, inner_loop_steps=1, batches_per_step=None,
        use_activation_checkpoint=False, samples_per_epoch=args.train_samples, noise_energy="bernoulli",
        wandb_entity=None, wandb_name="RLNN_lab", wandb_run_name=args.result_dir.name,
    )
    if args.method == "DiffUCO":
        settings.update(num_tp=5, num_kp=20, model="EncodeProcessDecodeTime1")
    vars(args).update(settings)
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    solver = (CAM if args.method == "CAM" else DiffUCO)(args)
    solver.prepare()
    if args.do_train:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        solver.train(args.save_dir)
        metadata = dict(version="v1", status="completed", mode="train", run=args.result_dir.name,
                        method=args.method, config=vars(args), started_at=started_at,
                        wall_time_s=perf_counter() - started)
        (args.result_dir / "run.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        return

    solver.model.eval()
    device = solver.accelerator.device
    params = sum(p.numel() for p in solver.model.parameters())
    decode = solver.problem.evaluate
    selected = None

    def record(graph, solution):
        scores, candidates = decode(graph, solution)
        nonlocal selected
        selected = candidates[:, scores.argmin()]
        return scores, candidates

    solver.problem.evaluate = record

    def solve(graph, generator, index):
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        with torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(args.seed + index)
            solver.evaluate_single(graph)
        return selected, {}

    evaluate(args, solve, device, params)


if __name__ == "__main__":
    main()
