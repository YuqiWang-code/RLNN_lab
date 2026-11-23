"""Train the official unweighted DGL-TreeSearch models on labeled DGL graphs."""
import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from prepare_supervised_baseline import prepare_dgl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dense-kernels", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    prepare_dgl(source)
    sys.path.insert(0, str(source.parent))
    from dgl_treesearch import train as upstream

    if args.dense_kernels:
        from dgl_dense_training import use_dense_training

        load_model = upstream._load_model

        def load_dense_model(*args, **kwargs):
            return use_dense_training(load_model(*args, **kwargs))

        upstream._load_model = load_dense_model

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    args.output.mkdir(parents=True)
    args.self_loops = True
    args.cuda_devices = [] if args.device == "cpu" else [args.device]
    args.model_prob_maps = 32
    args.pretrained_weights = None
    args.weighted = False
    args.lr = 0.001
    upstream.train(args)


if __name__ == "__main__":
    main()
