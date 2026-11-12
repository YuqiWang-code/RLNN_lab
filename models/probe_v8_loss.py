"""One-shot sanity check: is the A2/A3 loss calibrated at initialisation?

The student read-out is an exact copy of the frozen backbone at step 0, so the
marginal NLL on teacher blocks has to be close to the teacher's own entropy in
flip space.  A large value means the label convention, the mixture
normalisation, or the read-out copy is wrong -- not that training is hard.

    python -u models/probe_v8_loss.py --train_blocks outputs/v8_macro/data/blocks_train_ref0.pt \
        --backbone outputs/v2/none_seed0/checkpoints/best.pt --graphs 4
"""
import argparse

import torch

from models import GCN, edge2adj
from mis_v8_macro import MacroKernel, joint_log_likelihood, teacher_blocks, widen_readout
from v8_train import ARMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_blocks", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--graphs", type=int, default=4)
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blocks = torch.load(args.train_blocks, map_location="cpu", weights_only=False)[:args.graphs]
    teacher = torch.load(args.backbone, map_location="cpu", weights_only=True)
    net_ref = GCN(1, 128, 5, 1)
    net_ref.load_state_dict(teacher)
    net_ref = net_ref.to(device).eval()
    generator = torch.Generator().manual_seed(0)
    # Compare the widened copy against the frozen backbone on one real block: the
    # copy has to reproduce the teacher logits at initialisation.
    graph = torch.load(args.train_blocks, map_location="cpu", weights_only=False)[0]
    edge_index = graph["edge_index"].to(device)
    state = graph["blocks"][0]["state"].to(device).float()
    normalized = edge2adj(edge_index, state.size(0))
    with torch.no_grad():
        teacher_logits = net_ref(state.unsqueeze(-1), edge_index, normalized).squeeze(-1)
        kernel = MacroKernel(1).to(device)
        widened = widen_readout(teacher, 1, generator=generator)
        kernel.load_state_dict({k if k.startswith("net.") else "net." + k: v
                                for k, v in widened.items()})
        copy_logits = kernel(state, edge_index, normalized)[..., 0]
    print(f"teacher p mean {float(torch.sigmoid(teacher_logits).mean()):.4f} "
          f"absmax {float(teacher_logits.abs().max()):.2f} "
          f"copy diff {float((teacher_logits - copy_logits).abs().max()):.2e}")
    for arm in ("A2", "A3", "A5"):
        config = ARMS[arm]
        kernel = MacroKernel(config["components"]).to(device)
        widened = widen_readout(teacher, config["components"], generator=generator)
        kernel.load_state_dict({k if k.startswith("net.") else "net." + k: v
                                for k, v in widened.items()})
        kernel.eval()
        totals, nodes, flips, flips_b = [], 0, 0, 0
        with torch.no_grad():
            for graph in blocks:
                edge_index = graph["edge_index"].to(device)
                for block in graph["blocks"]:
                    state = block["state"].float().to(device)
                    y = block["continuation"].float().to(device)
                    b = block["record"].to(device).float() if config["record"] else None
                    adj = edge2adj(edge_index, state.size(0))
                    logits = kernel(state, edge_index, adj)
                    if arm == "A2":
                        print(f"  state shape {tuple(state.shape)} mean {float(state.mean()):.3f} "
                              f"edge_index {tuple(edge_index.shape)} max_index {int(edge_index.max())} "
                              f"adj rows {adj.size(0)}")
                    loss = joint_log_likelihood(logits, config["components"], state,
                                                (y, b), weight_record=config.get("weight_record", 1.0))
                    totals.append(float(loss) / state.size(0))
                    nodes += 1
                    flips += float(y.sum())
                    if b is not None:
                        flips_b += float(b.sum())
        print(f"{arm}: mean NLL/node={sum(totals)/len(totals):.4f} blocks={nodes} "
              f"flip_rate={flips/max(nodes,1)/ (state.size(0)):.5f} logit_absmax={float(logits.abs().max()):.3f}")


if __name__ == "__main__":
    main()
