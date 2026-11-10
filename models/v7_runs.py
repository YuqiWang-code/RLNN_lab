"""Shared paths and provenance for the static-context experiment."""

from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import torch

from eval_v2 import ROOT, TRAIN, VALID, TEST_CACHE, load_policy, policy_parser
from main import set_seed
from mis_v7_context import V7StaticContextGCN
from v7_structure import FEATURES


def backbone(ref):
    return ROOT / f"outputs/v2/none_seed{ref}/checkpoints/best.pt"


def pave(ref):
    return ROOT / f"outputs/v3/stats_h1_seed{ref}/checkpoints/best.pt"


def digest(path):
    with Path(path).open("rb") as stream:
        hashed = sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            hashed.update(block)
        return hashed.hexdigest()


def source_revision():
    if supplied := os.environ.get("RLNN_SOURCE_SHA"):
        return supplied
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise ValueError("Set RLNN_SOURCE_SHA to the synced revision before running")
    return result.stdout.strip()


def provenance(ref):
    try:
        mamba = version("mamba-ssm")
    except PackageNotFoundError:
        mamba = None
    return dict(source_sha=source_revision(), branch=os.environ.get("RLNN_SOURCE_BRANCH", "main"),
                backbone_weight_sha256=digest(backbone(ref)), pave_sha256=digest(pave(ref)),
                manifests={key: digest(folder / "tensor_manifest.json")
                           for key, folder in (("train", TRAIN), ("validation", VALID), ("test", TEST_CACHE))},
                static_feature_schema=FEATURES, mamba_version=mamba,
                torch_version=torch.__version__, cuda_version=torch.version.cuda)


def preflight_policy(mode, precision, *, ref=0, width=32, d_state=16, checkpoint=None, cuda_graph=False):
    parser = policy_parser()
    with TemporaryDirectory() as scratch:
        if mode == "none":
            folder = backbone(ref).parent
        elif checkpoint is not None:
            folder = Path(checkpoint)
        else:
            set_seed(42)
            model = V7StaticContextGCN(128, 5, mode, width, d_state, .5)
            missing, extra = model.load_state_dict(torch.load(backbone(ref), map_location="cpu", weights_only=True),
                                                   strict=False)
            assert not extra and all(name.startswith("adapter.") for name in missing)
            folder = Path(scratch)
            torch.save(model.state_dict(), folder / "best.pt")
        args = parser.parse_args(["--result_dir", scratch, "--save_dir", str(folder), "--split", "validation",
            "--context_mode", mode, "--mixed_precision", precision, "--v7_width", str(width),
            "--v7_d_state", str(d_state)])
        solver = load_policy(args)
        solver.cuda_graph_search = cuda_graph
        solver.model.requires_grad_(False)
    return solver
