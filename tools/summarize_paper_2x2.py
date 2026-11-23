"""Aggregate the paired CUDA Graph by compilation experiment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_mis_runs import read_rows, summarize


SEEDS = (1017, 1019, 1021)
ARMS = {
    "eager_no_compile": ("eager", False),
    "eager": ("eager", True),
    "no_compile": ("captured", False),
    "full": ("captured", True),
}
MODES = {"fixed130": 130, "deadline200": 0}


def load_mode(root, mode):
    keys = [(f"idx_{index}", str(seed), "0") for index in range(1000) for seed in SEEDS]
    arms = {}
    matrices = {}
    for arm, (backend, compiled) in ARMS.items():
        folder = root / mode / arm
        rows = read_rows(folder)
        if rows.keys() != set(keys):
            raise ValueError(f"{folder}: expected 1000 author graphs for each of the three seeds")
        expected_steps = MODES[mode]
        steps = [row[3] for row in rows.values()]
        if expected_steps and any(value != expected_steps for value in steps):
            raise ValueError(f"{folder}: fixed-step run diverged from 130 steps")
        if not expected_steps and any(row[2] is not None and row[2] > 200 for row in rows.values()):
            raise ValueError(f"{folder}: a late solution was scored")
        setups = {}
        eval_loops = {}
        wall_overruns = 0
        for seed in SEEDS:
            seed_dir = folder / f"batch_{seed}" / f"seed_{seed}"
            run = seed_dir / "padded" / "padded_ref0.json"
            data = json.loads(run.read_text(encoding="utf-8"))
            if (data["search_backend"], data["compile_model"], data["fixed_steps"],
                    data["num_k"], data["dtype"], data["dense"], data["pave"],
                    data["decode_config"]["captured"]["particles"],
                    data["budgets"], data["graphs"], data["seeds"],
                    data["checkpoint_dir"], data["test_path"],
                    data["leading_decode"], data["decode_schedule"]) != (
                    backend, compiled, expected_steps, 200, "fp16", 1, False, 4,
                    [0.2], 1000, [seed], "outputs/v2/none_seed0/checkpoints",
                    "models/data/MIS_tensors/RB_small_author_test", "off", "midpoint"):
                raise ValueError(f"{run}: configuration differs from the formal protocol")
            setups[str(seed)] = data["setup_s"]
            eval_loops[str(seed)] = data["wall_time_s"]
            with (seed_dir / "v11_rows.csv").open(encoding="utf-8", newline="") as stream:
                wall_overruns += sum(int(row["deadline_overrun"])
                                     for row in csv.DictReader(stream))
        arms[arm] = summarize(rows, 200)
        arms[arm]["setup_s_by_seed"] = setups
        arms[arm]["eval_loop_s_by_seed"] = eval_loops
        arms[arm]["by_seed"] = {
            str(seed): summarize({key: rows[key] for key in keys if key[1] == str(seed)}, 200)
            for seed in SEEDS
        }
        arms[arm]["wall_over_200ms"] = wall_overruns
        data = np.array([rows[key] for key in keys], dtype=float).reshape(1000, len(SEEDS), 4)
        matrices[arm] = data[:, :, [0, 1, 3]]
    return arms, matrices


def effects(matrices):
    eager, eager_compiled, captured, captured_compiled = (matrices[arm] for arm in ARMS)
    compile_eager = eager_compiled - eager
    compile_captured = captured_compiled - captured
    differences = {
        "capture_without_compile": captured - eager,
        "capture_with_compile": captured_compiled - eager_compiled,
        "compile_under_eager": compile_eager,
        "compile_under_capture": compile_captured,
        "capture_compile_interaction": compile_captured - compile_eager,
        "full_vs_eager_no_compile": captured_compiled - eager,
    }
    rng = np.random.default_rng(42)
    resamples = rng.integers(0, 1000, size=(5000, 1000))
    results = {}
    for name, diff in differences.items():
        per_graph = diff.mean(axis=1)
        draws = np.concatenate([per_graph[batch].mean(axis=1)
                                for batch in np.array_split(resamples, 10)])
        results[name] = {
            metric: {
                "mean": float(per_graph[:, index].mean()),
                "ci95": np.quantile(draws[:, index], [0.025, 0.975]).tolist(),
            }
            for index, metric in enumerate(("size", "wall_ms", "steps"))
        }
        size = diff[:, :, 0]
        results[name]["size_pairs"] = {
            "wins": int((size > 0).sum()), "ties": int((size == 0).sum()),
            "losses": int((size < 0).sum()), "max_absolute_delta": float(np.abs(size).max()),
            "mean_by_seed": dict(zip(map(str, SEEDS), size.mean(axis=0).tolist())),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = {}
    for mode in (args.mode,) if args.mode else MODES:
        arms, matrices = load_mode(args.root, mode)
        summary[mode] = {"arms": arms, "effects": effects(matrices)}
    result = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
