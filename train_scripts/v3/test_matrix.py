"""Run one GPU shard of the fixed-work or 0.5-second v3 test matrix."""
import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Job:
    index: int
    name: str
    backbone: str
    horizon: int
    seed: int
    posts: tuple
    cohort: str
    heads: dict = field(default_factory=dict)
    extended: bool = False
    reserve_s: float = 0.05


def backbone_checkpoint(backbone, seed):
    if backbone == "tuned":
        return ROOT / "outputs/v2" / f"none_seed{seed}" / "checkpoints"
    if seed == 0:
        return ROOT / "outputs/v000_rlnn_baseline/rb_train_author_b64_g8_gpu1_seed0/checkpoints"
    return ROOT / "outputs/v1" / f"rlnn_seed{seed}" / "checkpoints"


def head_checkpoint(output, mode, horizon, seed):
    return output / f"{mode}_h{horizon}_seed{seed}" / "checkpoints/best.pt"


def fixed_jobs(output):
    jobs = []
    for group, (backbone, horizon) in enumerate((("original", 1), ("tuned", 2), ("tuned", 1))):
        for seed in range(3):
            posts = ["none", "rule23", "random", "degree", "coverage", "pave"]
            heads = {"pave": head_checkpoint(output, "stats", horizon, seed)}
            if backbone == "tuned" and horizon == 1:
                for mode in ("message", "frequency"):
                    policy = "pave_" + mode
                    posts.append(policy)
                    heads[policy] = head_checkpoint(output, mode, 1, seed)
                adapted = head_checkpoint(output, "stats_adapted", 1, seed)
                if adapted.is_file():
                    posts.append("pave_adapted")
                    heads["pave_adapted"] = adapted
            jobs.append(Job(group * 3 + seed, f"test_{backbone}_h{horizon}_seed{seed}",
                            backbone, horizon, seed, tuple(posts), f"fixed_{backbone}_h{horizon}", heads))
    return jobs


def budget_jobs(output):
    selection = output / "budget_selection.json"
    reserve = float(json.loads(selection.read_text(encoding="utf-8")).get("reserve_s", 0.05)) if selection.is_file() else 0.05
    variants = (
        ("original_none_ext", "original", 1, "none", None, True, 0.0),
        ("tuned_none_ext", "tuned", 1, "none", None, True, 0.0),
        ("tuned_rule23_ext", "tuned", 2, "rule23", None, True, 0.02),
        ("tuned_random_h2", "tuned", 2, "random", None, False, reserve),
        ("tuned_stats_h1", "tuned", 1, "pave", "stats", False, reserve),
        ("tuned_message_h1", "tuned", 1, "pave_message", "message", False, reserve),
        ("tuned_frequency_h1", "tuned", 1, "pave_frequency", "frequency", False, reserve),
        ("tuned_adapted_h1", "tuned", 1, "pave_adapted", "stats_adapted", False, reserve),
    )
    jobs = []
    for seed in range(3):
        for offset, (variant, backbone, horizon, policy, mode, extended, reserve_s) in enumerate(variants):
            heads = {policy: head_checkpoint(output, mode, horizon, seed)} if mode else {}
            if mode == "stats_adapted" and not heads[policy].is_file():
                continue
            jobs.append(Job(seed * len(variants) + offset, f"budget_{variant}_seed{seed}",
                            backbone, horizon, seed, (policy,), f"budget_{variant}", heads,
                            extended=extended, reserve_s=reserve_s))
    return jobs


def build_command(job, stage, output, python):
    relative = output.relative_to(ROOT).as_posix()
    if stage == "fixed":
        baseline = f"{relative}/test_original_h1_seed{job.seed}/none"
        reference_name = f"test_tuned_h2_seed{job.seed}" if job.backbone == 'tuned' else job.name
        rule_reference = f"{relative}/{reference_name}/random"
    else:
        baseline = f"{relative}/budget_original_none_ext_seed{job.seed}/none"
        rule_reference = f"{relative}/budget_tuned_random_h2_seed{job.seed}/random"
    command = [python, "-u", str(ROOT / "models/eval_v3.py"),
               "--save_dir", str(backbone_checkpoint(job.backbone, job.seed)),
               "--result_dir", str(output / job.name), "--split", "test", "--test_samples", "1000",
               "--seed", "1017", "--train_seed", str(job.seed), "--backbone-seed", str(job.seed),
               "--head-seed", str(job.seed), "--num_t", "100", "--num_k", "20",
               "--frequency", "none", "--mixed_precision", "fp16",
               "--horizon", str(job.horizon), "--outer-rounds", "8",
               "--exchange_rounds", "3", "--repair-rounds", "4",
               "--candidate_cap", "32", "--improving-cap", "48",
               "--label", f"RLNN-{job.backbone}-h{job.horizon}", "--cohort-id", job.cohort,
               "--baseline-run", baseline, "--rule-reference-run", rule_reference,
               "--post", *job.posts]
    for policy, checkpoint in job.heads.items():
        if policy == "pave":
            command += ["--value-checkpoint", str(checkpoint)]
        else:
            command += ["--extra-value-checkpoint", policy, str(checkpoint)]
    if stage == "budget":
        command += ["--time-budget-s", "0.5", "--post-reserve-s", str(job.reserve_s)]
    if job.extended:
        command.append("--extended-search")
    return command


def completed(job, output):
    for policy in job.posts:
        path = output / job.name / policy / "run.json"
        if not path.is_file():
            return False
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("status") != "completed" or metadata.get("samples") != 1000:
            return False
    return True


def run_worker(stage, gpu, output):
    jobs = fixed_jobs(output) if stage == "fixed" else budget_jobs(output)
    jobs = [job for job in jobs if job.index % 2 == gpu]
    queue = output / "queue"
    queue.mkdir(parents=True, exist_ok=True)
    done = queue / f"{stage}_gpu{gpu}.done.txt"
    done.unlink(missing_ok=True)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", PYTORCH_ALLOC_CONF="backend:native")
    python = environment.get("PYTHON", sys.executable)
    taskset = shutil.which("taskset")
    for job in jobs:
        if completed(job, output):
            print(f"GPU {gpu}: already completed {job.name}", flush=True)
            continue
        for checkpoint in (backbone_checkpoint(job.backbone, job.seed) / "best.pt", *job.heads.values()):
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Required checkpoint for {job.name}: {checkpoint}")
        command = build_command(job, stage, output, python)
        if taskset:
            command = [taskset, "-c", f"{gpu * 24}-{gpu * 24 + 23}", *command]
        log_path = queue / f"{job.name}.log"
        print(f"GPU {gpu}: running {job.name}; log {log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n{datetime.now(timezone.utc).isoformat()}\n{shlex.join(command)}\n")
            log.flush()
            try:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                print(f"GPU {gpu}: {job.name} failed with exit {error.returncode}; log {log_path}",
                      file=sys.stderr, flush=True)
                raise SystemExit(error.returncode) from error
        if not completed(job, output):
            raise RuntimeError(f"{job.name} exited without all requested completed results; see {log_path}")
        print(f"GPU {gpu}: completed {job.name}", flush=True)
    done.write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    print(f"GPU {gpu}: {stage} shard completed ({len(jobs)} jobs)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fixed", "budget"), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/v3")
    args = parser.parse_args()
    run_worker(args.stage, args.gpu, args.output_root.resolve())


if __name__ == "__main__":
    main()
