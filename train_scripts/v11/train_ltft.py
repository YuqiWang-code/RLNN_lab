"""Run the upstream LTFT training with its RB-small configuration."""
import argparse
import os
from pathlib import Path
import sys

import dgl
from omegaconf import OmegaConf
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    upstream = args.source.resolve() / "gflownet"
    sys.path.insert(0, str(upstream))
    import data
    from main import main as train
    from util import seed_torch

    config = OmegaConf.load(upstream / "configs" / "main.yaml")
    config.d = args.device
    del config.defaults, config.hydra
    args.output.mkdir(parents=True)
    os.chdir(args.output)
    OmegaConf.save(config, "train_config.yaml")
    print(f"DGL {dgl.__version__}, PyTorch {torch.__version__}", flush=True)
    seed_torch(config.seed, verbose=False)

    prepare = data.imap_unordered_bar

    def prepare_with_workers(func, instances, n_processes=None):
        return prepare(func, instances, n_processes=8)

    data.imap_unordered_bar = prepare_with_workers
    train.__wrapped__(config)


if __name__ == "__main__":
    main()
