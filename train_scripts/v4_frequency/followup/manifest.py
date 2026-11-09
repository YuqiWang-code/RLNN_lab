"""Record fixed diagnostic inputs and check that inference preserves checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))
from probe_v4_utility import load_model
from mis_v3_value import load_pave_v3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    out = ROOT / "outputs/v4_frequency/followup"
    path = out / "manifest.json"
    if args.verify:
        manifest = json.loads(path.read_text())
        for item in manifest["checkpoints"]:
            assert digest(ROOT / item["path"]) == item["sha256"], item["path"]
        report = {"status": "passed", "checkpoint_count": len(manifest["checkpoints"]),
                  "checked_at": datetime.now(timezone.utc).isoformat()}
        (out / "p0_checks/checkpoints_unchanged.json").write_text(json.dumps(report, indent=2) + "\n")
        print(report)
        return
    rows = []
    for seed in range(3):
        for name, mode, relative in (
            ("B", "none", f"outputs/v2/none_seed{seed}/checkpoints/best.pt"),
            ("BFT", "none", f"outputs/v4_frequency/none_seed{seed}/checkpoints/best.pt"),
            ("MLP", "v4a_mlp", f"outputs/v4_frequency/v4a_mlp_seed{seed}/checkpoints/best.pt"),
            ("MSG", "v4a_message", f"outputs/v4_frequency/v4a_message_seed{seed}/checkpoints/best.pt"),
            ("F4", "v4a", f"outputs/v4_frequency/v4a_seed{seed}/checkpoints/best.pt"),
            ("PAVE", "stats", f"outputs/v3/stats_h1_seed{seed}/checkpoints/best.pt"),
        ):
            checkpoint = ROOT / relative
            if name == "PAVE":
                model = load_pave_v3(checkpoint, "cpu")
            else:
                model = load_model(mode, checkpoint, torch.device("cpu"))
            model.eval().requires_grad_(False)
            assert not model.training and not any(p.requires_grad for p in model.parameters())
            rows.append(dict(model=name, seed=seed, mode=mode, path=relative,
                             sha256=digest(checkpoint), bytes=checkpoint.stat().st_size,
                             params=sum(p.numel() for p in model.parameters()), keys=list(model.state_dict())))
    datasets = {}
    for split, directory, count in (("train", "RB_small_train", 20), ("validation", "RB_small_val", 100)):
        relative = f"models/data/MIS_tensors/{directory}"
        entries = json.loads((ROOT / relative / "tensor_manifest.json").read_text())["graphs"][:count]
        for item in entries:
            assert (ROOT / relative / item["tensor"]).is_file(), item
        datasets[split] = dict(path=relative, graphs=entries,
                               purpose="mechanism development" if split == "train" else "reused validation confirmation")
    record = dict(created_at=datetime.now(timezone.utc).isoformat(), source_sha=args.source_sha,
                  checkpoints=rows, datasets=datasets,
                  resources=json.loads((out / "p0_checks/resources.json").read_text()),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  fixed=dict(beta=1.02, K=20, T=100, d=5, layers=5, hidden=128,
                             search_seed=1017, precision_spectrum="fp32", precision_deploy="fp16",
                             time_budget_s=0.5, pave_reserve_s=0.05),
                  selection=dict(p1_train=20, p1_validation=100, p2_validation=100,
                                 p3_reliability_train=10, p3_node_groups_train=20,
                                 p3_validation="conditional on a useful sampling definition",
                                 official_test_used=False))
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"Recorded {len(rows)} frozen checkpoints and {sum(len(d['graphs']) for d in datasets.values())} graph IDs")


if __name__ == "__main__":
    main()
