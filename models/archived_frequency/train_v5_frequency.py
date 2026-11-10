"""Train v5 arms with the shared frozen RLNN teacher and original erdoes objective."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from main import build_parser, set_seed
from archived_frequency.mis_v5_frequency import V5_MODES
from solvers import RLNN
from archived_frequency.train_v4_frequency import MatchedStateRLNN


class V5TrainingRLNN(MatchedStateRLNN):
    @torch.no_grad()
    def generate_data(self, graph, A=None, normalized=None):
        states = super().generate_data(graph, A, normalized)
        if self.v5_frequency:
            self.probe = (states[:, :4].contiguous(), graph.edge_index, normalized)
        return states

    @torch.no_grad()
    def print_log(self, logger):
        logger.update(epoch=logger["Epoch"], step=logger["Iteration"], train_loss=logger["loss"],
                      train_expected_flips=logger.get("train_flips"),
                      val_size=None if logger["valid_obj"] is None else -logger["valid_obj"],
                      val_time=logger["valid_time"])
        model = self.accelerator.unwrap_model(self.model)
        if hasattr(self, "probe"):
            x, edges, topology = self.probe
            correction, details = model.adapter(x, topology, return_details=True)
            base = model.backbone_logits(x, edges, topology)
            gradients = [p.grad.square().sum() for p in model.adapter.parameters() if p.grad is not None]
            logger.update(alpha_low=float(details["a_low"]), alpha_high=float(details["a_high"]),
                          delta_logit_rms=float(correction.square().mean().sqrt()),
                          delta_prob_mean_abs=float(((base + correction).sigmoid() - base.sigmoid()).abs().mean()),
                          adapter_gradient_norm=float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0)
            for index, branch in enumerate(("low", "high")):
                gates = details["gates"][..., index].flatten()
                logger.update({f"gate_{branch}_mean": float(gates.mean()),
                               f"gate_{branch}_p10": float(gates.quantile(.1)),
                               f"gate_{branch}_p90": float(gates.quantile(.9))})
        RLNN.print_log(self, logger)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = build_parser()
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--state_source", choices=("teacher",), default="teacher")
    args = parser.parse_args()
    if args.frequency not in ("none",) + V5_MODES or not args.warm_start or args.resume:
        parser.error("v5 screening requires a v5 arm or none and a plain RLNN warm start")
    args.do_train = True
    run = args.result_dir
    run.mkdir(parents=True, exist_ok=False)
    args.save_dir = str(run / "checkpoints")
    Path(args.save_dir).mkdir()
    valid = json.loads((Path(args.valid_path) / "tensor_manifest.json").read_text(encoding="utf-8"))
    record = dict(version="v5_frequency", mode="train", status="running", seed=args.seed,
                  source_sha=os.environ.get("RLNN_SOURCE_SHA"), frequency=args.frequency, config=vars(args),
                  backbone_sha256=digest(args.warm_start),
                  train_manifest_sha256=digest(Path(args.train_path) / "tensor_manifest.json"),
                  validation_graph_ids=[Path(item["tensor"]).stem for item in valid["graphs"][:args.valid_samples]],
                  training_stream=20261 + args.seed, started_at=datetime.now(timezone.utc).isoformat())

    def save():
        (run / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    save()
    started = perf_counter()
    try:
        set_seed(args.seed)
        solver = V5TrainingRLNN(args)
        solver.prepare()
        set_seed(20261 + args.seed)
        record["params"] = sum(p.numel() for p in solver.model.parameters())
        record["adapter_params"] = sum(p.numel() for n, p in solver.model.named_parameters() if n.startswith("adapter."))
        record["device"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        record["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        save()
        solver.train(args.save_dir)
        state = torch.load(Path(args.save_dir) / "latest_training.pt", map_location="cpu", weights_only=False)
        history = state["history"]
        with (run / "training.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        record.update(status="completed", completed_epochs=state["epoch"], updates=history[-1]["Iteration"],
                      best_validation_size=-state["best_eval"], best_checkpoint_sha256=digest(Path(args.save_dir) / "best.pt"),
                      peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), train_wall_s=perf_counter()-started)
        save()


if __name__ == "__main__":
    main()
