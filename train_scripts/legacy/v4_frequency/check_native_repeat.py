"""Measure native forward and shared-random replay variation on real validation graphs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "models"))
import torch
from archived_frequency.diagnose_v4_branch_replay import load_group, replay, shared_noise
from eval_v2 import dataset_at, sync
from mis_v1 import adjacency
from archived_frequency.profile_v4_frequency import build_parser

parser = argparse.ArgumentParser()
parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--replay-graphs", type=int, default=3)
parser.add_argument("--output", type=Path)
cli = parser.parse_args()
records = []
def emit(record):
    records.append(record)
    print(json.dumps(record), flush=True)
args = build_parser().parse_args([
    "--seed", "0", "--split", "validation", "--graph-count", str(max(1, cli.replay_graphs)),
    "--precision", cli.precision, "--models", "baseline", "f4",
    "--result-dir", "outputs/v4_frequency/followup/p0_checks/native_repeat_probe",
])
torch.set_num_threads(1)
torch.manual_seed(17)
solvers, _ = load_group(args, args.models)
first = solvers["baseline"]
device = first.accelerator.device
dataset = dataset_at(args.test_path, first.dtype, args.graphs)


@torch.no_grad()
def run():
    for index, item in enumerate(dataset):
        graph = item.to(device)
        adj = adjacency(graph)
        topologies = {name: solver.policy_topology(graph, adj) for name, solver in solvers.items()}
        initial, uniforms, _ = shared_noise(graph.num_nodes, 20, 100, device, first.dtype, 1017 + index)
        if index == 0:
            state = initial.clone()
            for step in range(50):
                p = first.flip_logits(state, graph, adj, topologies["baseline"]).sigmoid()
                state = torch.where(uniforms[step] < p, 1 - state, state)
            for name, solver in solvers.items():
                values = []
                for repeat in range(cli.repeats):
                    value = solver.flip_logits(state, graph, adj, topologies[name]).detach().clone()
                    sync(device)
                    values.append(value)
                pairs = []
                for repeat in range(1, cli.repeats):
                    difference = (values[repeat] - values[0]).abs()
                    p0, p1 = values[0].sigmoid(), values[repeat].sigmoid()
                    pairs.append(dict(repeat=repeat, changed_logits=int((difference > 0).sum()),
                                      total_logits=difference.numel(), maximum_logit_difference=float(difference.max()),
                                      maximum_probability_difference=float((p1 - p0).abs().max()),
                                      sampled_flip_disagreements=int(((uniforms[50] < p0) != (uniforms[50] < p1)).sum())))
                emit(dict(kind="native_same_state", precision=cli.precision, model=name,
                          graph_id=Path(dataset.files[index]).stem, phase=50, comparisons=pairs))
        if not cli.replay_graphs:
            continue
        for name, solver in solvers.items():
            results = []
            for repeat in range(cli.repeats):
                result = replay(solver.problem, adj, graph.b, initial, uniforms,
                                lambda x: solver.flip_logits(x, graph, adj, topologies[name]),
                                phases=range(101), capture=True)
                sync(device)
                results.append(result)
            pairs = []
            for repeat in range(1, cli.repeats):
                changed = {step: int((results[0]["states"][step] != results[repeat]["states"][step]).sum())
                           for step in range(101)}
                pairs.append(dict(repeat=repeat, first_changed_step=next((step for step, count in changed.items() if count), None),
                                  changed_steps=sum(count > 0 for count in changed.values()),
                                  changed_state_entries=sum(changed.values()), final_state_changed=changed[100],
                                  selected_nodes_changed=results[0]["selected"] != results[repeat]["selected"]))
            emit(dict(kind="native_replay", precision=cli.precision, model=name,
                      graph_id=Path(dataset.files[index]).stem,
                      sizes=[result["selected"].bit_count() for result in results], comparisons=pairs))


run()

if cli.output:
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(dict(precision=cli.precision, repeats=cli.repeats, records=records), indent=2)+"\n", encoding="utf-8")
