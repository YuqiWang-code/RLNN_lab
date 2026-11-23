"""Repeat the four anytime budgets in a position- and predecessor-balanced order."""
import argparse
import csv
from contextlib import ExitStack
from ctypes import CDLL, c_double, c_int, c_uint
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))
import eval_online_mis_resident as online
import eval_priority_search as priority


SEEDS = (1017, 1019, 1021)
ARMS = ("learned", "degree", "online_mis")
BUDGETS = (50, 100, 200, 500)
CELLS = [(b, a) for b in BUDGETS for a in ARMS]
WILLIAMS = (0, 1, 11, 2, 10, 3, 9, 4, 8, 5, 7, 6)


def clock_time():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("models/data/MIS_tensors/ER_small_author_test"))
    p.add_argument("--output", type=Path, default=Path("outputs/v16/er_interleaved"))
    p.add_argument("--build", type=Path, default=Path("tmp/v16_anytime_build"))
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/v11/scale_priority_20261121/training/best.pt"))
    args = p.parse_args()
    entries = json.loads((args.dataset / "tensor_manifest.json").read_text())["graphs"]
    if len(entries) != 128:
        raise ValueError("Expected the full 128-graph ER test set")
    args.output.mkdir(parents=True, exist_ok=False)
    started = clock_time()
    print(f"{started} ER-small interleaved start", flush=True)
    torch.set_num_threads(1)
    params = argparse.Namespace(library=args.build / "priority_search.so", refine_ms=0,
                                refine_first=False, reserve_ms=3, budget_ms=200, degree_prior=True,
                                learned_mix=0.25, weight=0.7, plateau=50, perturb=4)
    priority_native = priority.load_native(params)
    scorer = priority.load_scorer(args.checkpoint)
    el = CDLL(str((args.build / "online_mis.so").resolve()))
    online_native = el.run_online_mis
    online_native.argtypes = [c_int, priority.IntPtr, priority.IntPtr, c_uint, c_double, priority.Emit]
    online_native.restype = None
    protocol = dict(started_at=started, host=platform.node(), torch=torch.__version__,
                    cuda_device=torch.cuda.get_device_name(0), gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                    cpu_affinity=sorted(os.sched_getaffinity(0)), checkpoint=str(args.checkpoint),
                    priority_schedule="ER_small", timing="Resident host edge list to copied CPU witnesses",
                    reserve_ms=3, seeds=SEEDS, cells=CELLS, williams_base_order=WILLIAMS,
                    indexing="graph in manifest order, then seed in listed order. Add block index modulo 12 to each base treatment index.",
                    graphs=len(entries), degree_prior=True, learned_mix=0.25, weight=0.7, plateau=50, perturb=4)
    protocol_path = args.output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    fields = ("graph_id", "seed", "repeat", "budget", "final_size", "wall_ms", "final_ready_ms",
              "first_ready_ms", "steps", "deadline_overrun", "has_solution", "prepare_ms", "score_ms",
              "upper_bound", "block", "period")
    with ExitStack() as stack:
        outputs, files = {}, []
        for budget, arm in CELLS:
            out = args.output / str(budget) / arm
            out.mkdir(parents=True)
            stream = stack.enter_context((out / "per_graph.csv").open("w", newline=""))
            archive = stack.enter_context((out / "solutions.jsonl").open("w"))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            outputs[budget, arm] = writer, archive
            files.extend((stream, archive))
        for index, entry in enumerate(entries):
            raw = torch.load(args.dataset / entry["tensor"], weights_only=True, map_location="cpu")
            graph = Path(entry["source"]).stem
            for s, seed in enumerate(SEEDS):
                block = len(SEEDS) * index + s
                for period, offset in enumerate(WILLIAMS):
                    budget, arm = CELLS[(block + offset) % len(CELLS)]
                    if arm == "online_mis":
                        wall, events = online.solve(raw, seed, online_native, budget, params.reserve_ms)
                        steps = prep = score = upper = None
                    else:
                        params.budget_ms = budget
                        wall, events, steps, prep, score, upper = priority.solve(
                            raw, seed, params, *priority_native, scorer if arm == "learned" else None)
                    writer, archive = outputs[budget, arm]
                    writer.writerow(dict(graph_id=graph, seed=seed, repeat=0, budget=budget/1000,
                                         final_size=len(events[-1][1]) if events else 0, wall_ms=wall,
                                         final_ready_ms=events[-1][0] if events else None,
                                         first_ready_ms=events[0][0] if events else None, steps=steps,
                                         deadline_overrun=int(wall > budget), has_solution=int(bool(events)),
                                         prepare_ms=prep, score_ms=score, upper_bound=upper,
                                         block=block, period=period))
                    archive.write(json.dumps(dict(graph_id=graph, seed=seed, updates=[
                        dict(ready_ms=t, nodes=nodes.tolist()) for t, nodes in events]),
                        separators=(",", ":")) + "\n")
            if (index + 1) % 8 == 0:
                for stream in files:
                    stream.flush()
                print(f"{clock_time()} {index+1}/{len(entries)} graphs", flush=True)
    protocol["finished_at"] = clock_time()
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"{protocol['finished_at']} ER-small interleaved done", flush=True)


if __name__ == "__main__":
    main()
