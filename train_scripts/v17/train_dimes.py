"""Retrain the released DIMES-MIS meta-solver on the independent ER training split."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.io import savemat
from scipy.sparse import csr_matrix
import torch


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "reference_methods/DIMES/source/MIS/solvers/intel_treesearch/NPHard"
WORK = ROOT / "tmp/dimes_er_retrain"
OUT = ROOT / "outputs/v17/dimes_training"
TRAIN = ROOT / "models/data/scale_priority_20261121/ER_small/train"
SEED = 32871


def prepare_mat_inputs():
    inputs = WORK / "mat"
    inputs.mkdir(parents=True, exist_ok=True)
    records = json.loads((TRAIN / "tensor_manifest.json").read_text())["graphs"]
    if len(records) != 128:
        raise ValueError("Expected the 128 independent small ER training graphs")
    for record in records:
        graph = torch.load(TRAIN / record["tensor"], map_location="cpu", weights_only=True)
        src, dst = graph["edge_index"].numpy()
        n = int(graph["num_nodes"])
        matrix = csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(n, n))
        savemat(inputs / (record["graph_id"] + ".mat"), {"adj": matrix})
    return inputs


def stage_trainer():
    stage = WORK / "upstream"
    shutil.copytree(UPSTREAM / "gcn", stage / "gcn", dirs_exist_ok=True)
    for path, changes in {
        stage / "gcn/utils.py": {
            "scipy.sparse.linalg.eigen.arpack import": "scipy.sparse.linalg import",
            "dtype=np.bool)": "dtype=bool)",
        },
        stage / "gcn/models.py": {"import tensorflow_addons as tfa\n": ""},
        stage / "gcn/reinforce.py": {"np.random.randint(sys.maxsize)": "np.random.randint(2**31 - 1)"},
    }.items():
        code = path.read_text()
        for old, new in changes.items():
            code = code.replace(old, new)
        path.write_text(code)
    script = stage / "train_dimes_batched.py"
    code = (UPSTREAM / script.name).read_text()
    code = code.replace("from torch.utils.tensorboard import SummaryWriter", "")
    code = code.replace("torch_writer = SummaryWriter(train_log_dir + '_scalar')", "")
    code = code.replace("torch_writer.add_scalar(key, results, step)", "print(key, results, step)")
    code = code.replace("torch_writer.flush()", "")
    code = code.replace("reinforce.beam_size = 1024", "reinforce.beam_size = 32")
    code = code.replace("tf.disable_v2_behavior()", f"tf.disable_v2_behavior()\ntf.set_random_seed({SEED})\nnp.random.seed({SEED})\nrandom.seed({SEED})")
    code = code.replace("train_mat_names = os.listdir(data_path)", "train_mat_names = sorted(os.listdir(data_path))")
    script.write_text(code)
    return script


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = prepare_mat_inputs()
    script = stage_trainer()
    command = [sys.executable, "-u", str(script), str(inputs), str(OUT), "--self_loops",
               "--normalize_factor", "1.0", "--meta_update", "--meta_steps", "1",
               "--epochs", "4", "--model_prob_maps", "1", "--batch_size", "8"]
    protocol = dict(revision="95aaa91c74cd47dc441c84eea9d38ead44f61ffd",
                    training_split=TRAIN.relative_to(ROOT).as_posix(), graphs=128, epochs=4,
                    minibatch=8, updates=64, heatmaps=1, rollouts=32, meta_steps=1,
                    seed=SEED, selection="last epoch fixed before test", command=command)
    (OUT / "run.json").write_text(json.dumps(protocol, indent=2) + "\n")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
