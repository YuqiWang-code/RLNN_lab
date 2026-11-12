"""Train one v8 ACMD arm on frozen-teacher macro blocks.

Arms (first round, §11.3): ``A2`` M=1 endpoint only, ``A3`` M=2 endpoint only,
``A5`` M=2 endpoint + record.  ``A4``/``A6`` exist for the second round and reuse
the same code path.  The objective is the joint likelihood over the mixture; the
``marginal`` switch is the A6 control (average the components first).

    python -u models/v8_train.py --arm A5 --reference_seed 0 \
        --train_blocks outputs/v8_macro/data/blocks_train_ref0.pt \
        --save_dir outputs/v8_macro/train_A5_ref0 --epochs 3
"""
import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from models import edge2adj
from mis_v8_macro import MacroKernel, joint_log_likelihood, marginal_bce, widen_readout
from v7_runs import digest


ARMS = dict(
    A2=dict(components=1, record=False, marginal=False),
    A3=dict(components=2, record=False, marginal=False),
    A4=dict(components=1, record=True, marginal=False),
    A5=dict(components=2, record=True, marginal=False),
    A6=dict(components=2, record=True, marginal=True),
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--reference_seed", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--train_blocks", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_blocks_per_graph", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=200)
    return parser


def main():
    args = build_parser().parse_args()
    config = ARMS[args.arm]
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blocks = torch.load(args.train_blocks, map_location="cpu", weights_only=False)
    teacher = torch.load(args.backbone, map_location="cpu", weights_only=True)
    generator = torch.Generator().manual_seed(args.seed)
    kernel = MacroKernel(config["components"]).to(device)
    # The frozen backbone checkpoint stores the plain GCN keys; MacroKernel keeps the
    # backbone under ``net``.  Widen first (it expects the flat names), then prefix.
    widened = widen_readout(teacher, config["components"], generator=generator)
    kernel.load_state_dict({key if key.startswith("net.") else "net." + key: value
                            for key, value in widened.items()})
    kernel.train()
    optimiser = torch.optim.Adam(kernel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        losses, seen = [], 0
        for graph in blocks:
            edge_index = graph["edge_index"].to(device)
            size = graph["blocks"][0]["state"].size(0)
            # The frozen backbone reads the symmetrically normalised adjacency with
            # self loops; feeding it the raw 0/1 adjacency silently breaks the
            # teacher copy (the initial NLL jumps from ~0.03 to ~0.7 per element).
            adj = edge2adj(edge_index, size)
            for block in graph["blocks"][:args.max_blocks_per_graph]:
                state = block["state"].to(device).float()
                y = block["continuation"].to(device).float()
                b = block["record"].to(device).float() if config["record"] else None
                logits = kernel(state, edge_index, adj)
                if config["marginal"]:
                    loss = marginal_bce(logits, config["components"], (y, b))
                else:
                    loss = joint_log_likelihood(logits, config["components"], state, (y, b))
                nodes = state.numel()
                loss = loss / nodes
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                losses.append(float(loss))
                seen += 1
                if args.log_every and seen % args.log_every == 0:
                    print(f"epoch {epoch} step {seen} loss {mean(losses[-args.log_every:]):.5f}", flush=True)
        history.append(dict(epoch=epoch, steps=seen, loss=mean(losses)))
        print(f"epoch {epoch}: {seen} blocks, loss {mean(losses):.5f}", flush=True)
    torch.save(dict(state_dict=kernel.state_dict(), arm=args.arm, components=config["components"],
                    record=config["record"], marginal=config["marginal"], history=history),
               args.save_dir / "best.pt")
    (args.save_dir / "run.json").write_text(json.dumps(
        dict(version="v8_macro", mode="train", arm=args.arm, ref=args.reference_seed,
             config=vars(args), arms=config, history=history,
             graphs=len(blocks), blocks=sum(len(g["blocks"]) for g in blocks),
             backbone_sha256=digest(args.backbone)), indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8")
    print(f"wrote {args.save_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
