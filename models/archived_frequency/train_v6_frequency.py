"""Train SC-RLNN ablations on matched trajectories from Backbone B."""

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from time import perf_counter

import torch

from eval_v2 import write_run
from main import build_parser, set_seed
from archived_frequency.mis_v6_frequency import V6_MODES
from archived_frequency.train_v4_frequency import MatchedStateRLNN


def main():
    parser = build_parser()
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--teacher_checkpoint", type=Path, required=True)
    parser.add_argument("--lambda_dir", type=float, default=0.03)
    parser.add_argument("--lambda_ortho", type=float, default=0.01)
    args = parser.parse_args()
    if args.frequency not in V6_MODES or args.warm_start:
        parser.error("Select a v6 arm and a separate frozen teacher checkpoint")
    args.do_train = True
    run = args.result_dir
    run.mkdir(parents=True, exist_ok=False)
    args.save_dir = str(run / "checkpoints")
    Path(args.save_dir).mkdir()
    record = dict(version="v6", mode="train", status="running", seed=args.seed,
                  frequency=args.frequency, config=vars(args), teacher=args.teacher_checkpoint,
                  training_stream=20261 + args.seed, source_sha=os.environ.get("RLNN_SOURCE_SHA"),
                  gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), started_at=datetime.now(timezone.utc).isoformat())
    write_run(run, record)
    started = perf_counter()
    try:
        set_seed(args.seed)
        solver = MatchedStateRLNN(args, teacher_checkpoint=args.teacher_checkpoint)
        solver.prepare()
        set_seed(record["training_stream"])
        record["params"] = sum(p.numel() for p in solver.model.parameters())
        solver.train(args.save_dir)
        state = torch.load(Path(args.save_dir) / "latest_training.pt", map_location="cpu", weights_only=False)
        history = state["history"]
        with (run / "training.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        record.update(status="completed", completed_epochs=state["epoch"], updates=history[-1]["Iteration"],
                      best_validation_size=-state["best_eval"],
                      peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), train_wall_s=perf_counter()-started)
        write_run(run, record)


if __name__ == "__main__":
    main()
