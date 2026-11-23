"""Copy the upstream LwD policy and environment for current DGL and PyTorch."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    shutil.copytree(args.source / "ppo", args.destination / "ppo", dirs_exist_ok=True)
    shutil.copy2(args.source / "env.py", args.destination / "env.py")
    for path in [args.destination / "env.py", args.destination / "ppo" / "graph_net.py",
                 args.destination / "ppo" / "actor_critic.py"]:
        code = path.read_text().replace("fn.copy_src(src=", "fn.copy_u(u=")
        code = code.replace("torch.LongTensor(\n            self.g.batch_num_nodes()\n            ).to(self.device)",
                            "self.g.batch_num_nodes().to(self.device, dtype=torch.long)")
        code = code.replace("node_mask.view(-1)", "node_mask.reshape(-1)")
        path.write_text(code)


if __name__ == "__main__":
    main()
