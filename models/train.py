import os
import json
from pathlib import Path
from argparse import ArgumentParser, SUPPRESS

import torch
import wandb

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies.ddp import DDPStrategy

from diffusion.pl_tsp_model import TSPModel
from diffusion.pl_mis_model import MISModel
from diffusion.pl_cvrp_model import CVRPModel

import warnings
warnings.filterwarnings("ignore")

MODEL_REGISTRY = {
    "tsp": (TSPModel, "min"),
    "mis": (MISModel, "max"),
    "cvrp": (CVRPModel, "min"),
}


@rank_zero_only
def save_run_config(args):
    destination = Path(args.storage_path) / 'run.json'
    destination.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding='utf-8')


def arg_parser():
    parser = ArgumentParser(description="Train or test a CO diffusion model.")
    parser.add_argument("--task", choices=MODEL_REGISTRY.keys(), required=True)
    parser.add_argument("--model_type", choices=["gcn", "transformer"], default="gcn")
    parser.add_argument("--storage_path", required=True)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--training_split")
    parser.add_argument("--training_split_label_dir")
    parser.add_argument("--validation_split")
    parser.add_argument("--validation_split_label_dir")
    parser.add_argument("--test_split")
    parser.add_argument("--test_split_label_dir")
    parser.add_argument("--validation_examples", type=int, default=64)
    parser.add_argument("--graph_type", choices=["undirected", "directed"], default="undirected")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_activation_checkpoint", action="store_true")

    parser.add_argument("--diffusion_schedule", default="linear")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_diffusion_steps", type=int, default=1, help=SUPPRESS)
    parser.add_argument("--inference_schedule", default="cosine")
    parser.add_argument("--sequential_sampling", type=int, default=1)
    parser.add_argument("--parallel_sampling", type=int, default=1)

    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sparse_factor", type=int, default=-1)
    parser.add_argument("--aggregation", default="sum")
    parser.add_argument("--two_opt_iterations", type=int, default=0)
    parser.add_argument("--save_numpy_heatmap", action="store_true")
    parser.add_argument("--save_tsp_rewrite_trace", action="store_true")
    parser.add_argument("--tsp_rewrite_trace_dir")

    parser.add_argument("--project_name", default="co_diffusion")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_logger_name")
    parser.add_argument("--resume_id")
    parser.add_argument("--ckpt_path")
    parser.add_argument("--resume_weight_only", action="store_true")

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--rewrite_ratio", type=float, default=0.25)
    parser.add_argument("--norm", action="store_true", default=False)
    parser.add_argument("--rewrite", action="store_true", default=True, help=SUPPRESS)
    parser.add_argument("--rewrite_steps", type=int, default=1)
    parser.add_argument("--rewrite_iterations", type=int, default=1)
    parser.add_argument("--guided", action="store_true")

    parser.add_argument("--consistency", action="store_true")
    parser.add_argument("--boundary_func", default="truncate")
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--c1", type=float, default=50)
    parser.add_argument("--c2", type=float, default=50)
    parser.add_argument("--m1", type=float, default=0.0)
    parser.add_argument("--m2", type=float, default=0.0)
    parser.add_argument("--ep", type=float, default=0.0)
    parser.add_argument("--offline", action="store_true")

    return parser.parse_args()


def main(args):
    seed_everything(args.seed, workers=True)
    if args.offline or ("WANDB_MODE" in os.environ and os.environ["WANDB_MODE"] == "offline"):
        os.environ["WANDB_MODE"] = "offline"
        wandb.init()
    else:
        wandb.login(key=os.environ["WANDB_API_KEY"])

    model_class, saving_mode = MODEL_REGISTRY[args.task]
    model = model_class(param_args=args)
    os.makedirs(args.storage_path, exist_ok=True)

    wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
    wandb_logger = WandbLogger(
        name=args.wandb_logger_name,
        project=args.project_name,
        entity=args.wandb_entity,
        save_dir=args.storage_path,
        id=args.resume_id or wandb_id,
    )
    csv_logger = CSVLogger(save_dir=args.storage_path, name='metrics', version='')
    rank_zero_info(
        f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}"
    )
    rank_zero_info(args)

    run_name = args.wandb_logger_name or wandb_logger.name or args.project_name
    checkpoint_callback = ModelCheckpoint(
        monitor="val/solved_cost",
        mode=saving_mode,
        save_top_k=1,
        save_last=True,
        dirpath=os.path.join(
            wandb_logger.save_dir,
            run_name,
            wandb_logger._id,
            "checkpoints",
        ),
    )
    callbacks = [
        TQDMProgressBar(refresh_rate=1),
        checkpoint_callback,
        LearningRateMonitor(logging_interval="step"),
    ]
    gpu_count = torch.cuda.device_count()
    trainer = Trainer(
        accelerator="gpu" if gpu_count else "cpu",
        devices=gpu_count or 1,
        max_epochs=args.num_epochs,
        callbacks=callbacks,
        logger=[wandb_logger, csv_logger],
        check_val_every_n_epoch=1,
        strategy=DDPStrategy(static_graph=True) if gpu_count > 1 else "auto",
        precision=16 if args.fp16 else 32,
        inference_mode=False,
    )
    save_run_config(args)

    ckpt_path = args.ckpt_path
    if args.do_train:
        if args.do_test:
            trainer.test(model)
        if args.resume_weight_only:
            model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=ckpt_path)
    elif args.do_test:
        trainer.test(model, ckpt_path=ckpt_path)
    for logger in trainer.loggers:
        logger.finalize("success")


if __name__ == "__main__":
    args = arg_parser()
    main(args)
