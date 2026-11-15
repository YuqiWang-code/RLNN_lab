"""Paired static-bias interventions on the locked RB validation cohort."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch

from eval_v2 import ROOT, VALID, dataset_at, search, sync, write_run
from mis_static_prior import bias_stats, oracle_bias, random_bias
from mis_v1 import adjacency
from mis_v3_search import SearchGraph, node_mask, nodes, platform_search
from mis_v3_value import load_pave_v3
from static_prior_protocol import ORACLE_NAMES, P0_ARMS
from v7_runs import backbone, digest, pave, preflight_policy, provenance
from v7_structure import V7Topology


STEPS = (25, 50, 100, 200, 400, 800)
BUDGETS = (.5, 1.)
RANDOM_RMS = .0107


def search_seeds(index, count):
    return [1017 + index + 100_003 * replica for replica in range(count)]


def cohort(args):
    dtype = torch.float32 if args.backend == "fp32-graph" else torch.float16
    dataset = dataset_at(VALID, dtype)
    dataset.files = dataset.files[args.offset:args.offset + args.samples]
    if len(dataset) != args.samples:
        raise ValueError("The requested validation cohort is incomplete")
    return dataset, [Path(path).stem for path in dataset.files]


def add_common(parser):
    parser.add_argument("--ref", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seeds-per-graph", type=int, default=3)
    parser.add_argument("--backend", choices=("fp32-graph", "fp16-eager"), default="fp32-graph")
    parser.add_argument("--checkpoint", type=Path, help="Training directory or checkpoints directory")
    parser.add_argument("--oracle-kamis-path", "--static_oracle_path", dest="oracle_kamis_path", type=Path)
    parser.add_argument("--oracle-b-path", type=Path)
    parser.add_argument("--oracle-amplitude", type=float, default=1., help="Amplitude for oracle arms without @A")
    parser.add_argument("--static-bias-seed", "--static_bias_seed", type=int, default=7311)
    parser.add_argument("--arms", nargs="+", help="b0, null-rand, amp@A, rand@A, oracle-kamis@A, oracle-b@A, anti-oracle@A, "
                        "centered, v1-node-only, masked, matched-rand, or a label supplied by --compare")
    parser.add_argument("--compare", nargs=2, action="append", metavar=("LABEL", "RUN"), default=[])
    parser.add_argument("--feature-drop", choices=("log_degree", "neighbor_log_degree_mean",
                        "neighbor_log_degree_std", "core", "hub_distance", "two_hop_fraction", "density", "size"))


def checkpoint_paths(path, ref):
    if path is None:
        path = ROOT / f"outputs/static_prior_v1/train_centered_ref{ref}"
    folder = path if (path / "best.pt").is_file() else path / "checkpoints"
    return folder, folder.parent / "run.json"


def load_arm_policy(args, path):
    folder, record_path = checkpoint_paths(path, args.ref)
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    config = record.get("config", {})
    mode = config.get("context_mode", "static_centered")
    if mode not in ("static_centered", "static_constants"):
        raise ValueError(f"{folder} must contain a centered or constants checkpoint")
    options = []
    for name in ("static_constant_limit", "static_node_limit", "static_features", "static_encoder",
                 "v7_delta_limit", "v7_order_seed", "v7_order_views"):
        if config.get(name) is not None and not (name == "v7_delta_limit" and record.get("version") == "static_prior_v1"):
            options.extend((f"--{name}", str(config[name])))
    if config.get("static_peak_normalize"):
        options.append("--static_peak_normalize")
    policy = preflight_policy(mode, "no" if args.backend == "fp32-graph" else "fp16", ref=args.ref,
                             width=config.get("v7_width", 32), d_state=config.get("v7_d_state", 16),
                             checkpoint=folder, cuda_graph=args.backend == "fp32-graph", options=options)
    return policy, folder / "best.pt", record


def baseline(args):
    return preflight_policy("none", "no" if args.backend == "fp32-graph" else "fp16", ref=args.ref,
                            cuda_graph=args.backend == "fp32-graph")


class Experiment:
    def __init__(self, args, graph_ids):
        self.args = args
        self.arms = args.arms or P0_ARMS
        self.base = baseline(args)
        self.device = self.base.accelerator.device
        self.policies = {"b0": self.base}
        self.records = {}
        self.paths = {"b0": backbone(args.ref)}
        self.targets = {}
        self.target_hashes = {}
        self.norm_hashes = {}
        self.alpha = {}
        self.compares = dict(args.compare)
        self.doses = {}
        target_files = {}
        if len(self.arms) != len(set(self.arms)) or "b0" not in self.arms:
            raise ValueError("Use distinct arms including b0")
        if any(arm != "b0" and arm not in self.compares for arm in self.arms):
            self.adapter, self.adapter_path, self.adapter_record = load_arm_policy(args, args.checkpoint)
        for arm in self.arms:
            if arm == "b0":
                continue
            name, separator, dose = arm.partition("@")
            if arm in self.compares:
                checkpoint, _ = checkpoint_paths(Path(self.compares[arm]), args.ref)
                if hasattr(self, "adapter") and checkpoint.resolve() == self.adapter_path.parent.resolve():
                    policy, path, record = self.adapter, self.adapter_path, self.adapter_record
                else:
                    policy, path, record = load_arm_policy(args, Path(self.compares[arm]))
            else:
                policy, path, record = self.adapter, self.adapter_path, self.adapter_record
            self.policies[arm], self.paths[arm], self.records[arm] = policy, path, record
            self.norm_hashes[arm] = record.get("feature_norm_sha256")
            alpha = getattr(policy.accelerator.unwrap_model(policy.model).adapter, "alpha", None)
            if alpha is not None:
                self.alpha[arm] = float(alpha)
            if name in ORACLE_NAMES:
                path = args.oracle_b_path if name == "oracle-b" else args.oracle_kamis_path
                if path is None:
                    raise ValueError(f"{arm} needs a target JSON from prepare_oracles.sh")
                if path not in target_files:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    missing = set(graph_ids) - data.keys()
                    if missing:
                        raise ValueError(f"{path} lacks target for {min(missing)}")
                    target_files[path] = data, digest(path)
                self.targets[arm], self.target_hashes[arm] = target_files[path]
                self.doses[arm] = float(dose) if separator else args.oracle_amplitude
            elif name in ("amp", "rand") and separator:
                self.doses[arm] = float(dose)
            elif arm not in self.compares and arm not in ("null-rand", "centered", "v1-node-only", "masked", "matched-rand"):
                raise ValueError(f"Unknown arm {arm}")
            if arm in self.doses and not (0 <= self.doses[arm] < math.inf):
                raise ValueError(f"Invalid amplitude in {arm}")
        original = self.base.accelerator.unwrap_model(self.base.model).state_dict()
        for policy in set(self.policies.values()) - {self.base}:
            weights = policy.accelerator.unwrap_model(policy.model).state_dict()
            if any(not torch.equal(value, weights[key]) for key, value in original.items()):
                raise ValueError("All paired adapters must use the frozen backbone for --ref")

    def policy(self, arm, graph_id, graph_index, legal, steps):
        solver = self.policies[arm]
        solver.num_t = steps
        if arm == "b0":
            return solver
        model = solver.accelerator.unwrap_model(solver.model)
        adapter = model.adapter
        name = arm.partition("@")[0]
        target = self.targets[arm][graph_id] if arm in self.targets else None
        if arm in self.targets:
            if not isinstance(target, list) or not all(isinstance(i, int) and 0 <= i < legal.num_nodes for i in target):
                raise ValueError(f"Invalid target nodes for {graph_id} in {arm}")
            if not legal.feasible(node_mask(target)):
                raise ValueError(f"Conflicting {name} target on {graph_id}; regenerate the oracle targets")
        amplitude = self.doses.get(arm)
        drop = self.args.feature_drop if solver is getattr(self, "adapter", None) else None
        adapter.feature_drop = adapter.feature_schema.index(drop) if drop else None
        adapter.centered = adapter.mode == "static_centered" and (
            arm in ("centered", "v1-node-only", "matched-rand") or arm in self.compares
            or (name in ("amp", "rand") and amplitude > 0))

        def bias(adapter, topology, graph):
            device = graph.edge_index.device
            seed = self.args.static_bias_seed + graph_index
            if target is not None:
                direction = oracle_bias(graph.num_nodes, target, amplitude, device)
                return -direction if name == "anti-oracle" else direction
            if arm == "null-rand":
                return random_bias(graph.num_nodes, seed, RANDOM_RMS, device)
            if name in ("amp", "rand"):
                if amplitude == 0:
                    return torch.zeros((1, 2), device=device)
                direction = adapter.direction(topology, normalize=True)
                sign = torch.where(adapter.alpha < 0, -1., 1.)
                scaled = (amplitude * sign) * direction
                if name == "rand":
                    return random_bias(graph.num_nodes, seed, scaled.square().mean().sqrt(), device)
                return scaled
            if arm == "v1-node-only":
                return adapter.action_bias(topology, (0., model.delta_limit[1]))
            if arm == "matched-rand":
                constants = adapter.constant_limit * adapter.constants.tanh()[None, :]
                learned = adapter.action_bias(topology, model.delta_limit) - constants
                rms = learned.square().mean().sqrt()
                return constants + random_bias(graph.num_nodes, seed, rms, device)
            return adapter.action_bias(topology, model.delta_limit)

        solver.static_intervention = bias
        return solver


def initial_state(graph, solver, seed):
    rng = torch.Generator(device=graph.b.device).manual_seed(seed)
    x = torch.randint(0, 2, (graph.num_nodes, solver.num_k), dtype=solver.dtype,
                      device=graph.b.device, generator=rng)
    return x, rng


def probability_metrics(p, base, noise):
    flip = noise < p
    base_flip = noise < base
    return dict(delta_prob_mae=float((p - base).float().abs().mean()),
                boundary_fraction=float(((p > .01) & (p < .99)).float().mean()),
                flip_rate=float(p.float().sum(0).mean()),
                base_flip_rate=float(base.float().sum(0).mean()),
                flip_disagreement=float((flip != base_flip).float().mean()))


def measure_initial(experiment, graph, seed, prepared):
    adj = adjacency(graph)
    x, rng = initial_state(graph, experiment.base, seed)
    base_topology = experiment.base.policy_topology(graph, adj)
    base = experiment.base.flip_logits(x, graph, adj, base_topology).sigmoid()
    noise = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=rng)
    stats = {}
    for key, bias in prepared.items():
        arm, _ = key
        if bias is None:
            bias, p = x.new_zeros((1, 2)), base
        else:
            bias = bias.to(x.device)
            topology = V7Topology(base_topology, None, (), (), 1, bias)
            p = experiment.policies[arm].flip_logits(x, graph, adj, topology).sigmoid()
        stats[key] = dict(**bias_stats(bias), **probability_metrics(p, base, noise))
    return stats


def solve_rows(solver, graph, edges, seed, ranker, steps, budget):
    device = graph.b.device
    solver.num_t = max(steps) if budget is None else 10000
    sync(device)
    resident = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    deadline = start + budget if budget else None
    legal = SearchGraph(graph.num_nodes, edges, seed=seed)
    timings, snapshots, curve = {}, {}, []
    bias = None

    def remember(topology):
        nonlocal bias
        bias = topology.cached_action_bias

    def collect(step, selected, incumbent, periodic):
        if periodic:
            curve.append((step, incumbent.bit_count()))
        if step in steps:
            snapshots[step] = selected, incumbent, perf_counter() - start

    selected, used = search(solver, graph, legal, seed, deadline=deadline-.05 if budget else None,
                            decode_every=10, timings=timings, checkpoints=steps if budget is None else (),
                            on_decode=collect if budget is None else None,
                            on_prepare=remember if solver.v7_context else None, archive_final=budget is not None)
    sync(device)
    search_time = perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0

    def finish(mask, used, time, protocol, decoder, reached):
        tick = perf_counter()
        fallback = mask == 0 and graph.num_nodes > 0
        if fallback:
            mask = 1 << legal.greedy_order[0]
        final = platform_search(legal, mask, choose=ranker.choose, horizon=1,
                                deadline=deadline if budget else float("inf")).best
        end = perf_counter()
        if not legal.feasible(mask) or not legal.feasible(final):
            raise ValueError("Search returned an infeasible independent set")
        return dict(protocol=protocol, decoder=decoder, base_size=mask.bit_count(), final_size=final.bit_count(),
                    search_steps=used, search_s=time, pave_s=end-tick, solve_s=end-start if budget else time+end-tick,
                    prepare_budget_fraction=timings["policy_prepare_s"]/budget if budget else None,
                    **timings, feasible=1, fallback=int(fallback), overrun=int(budget is not None and end > deadline),
                    peak_memory_mib=peak/2**20, solve_peak_mib=(peak-resident)/2**20, steps_to_target=reached,
                    base_nodes=" ".join(map(str, nodes(mask))), selected_nodes=" ".join(map(str, nodes(final))))

    if budget:
        rows = [finish(selected, used, search_time, f"budget{int(budget * 10):02d}", "periodic", None)]
    else:
        target = snapshots[max(steps)][0].bit_count()
        samples = curve + [(step, row[0].bit_count()) for step, row in snapshots.items()]
        reached = min(step for step, size in samples if size >= target)
        rows = [finish(mask, step, time, f"t{step}", decoder, reached)
                for step, (endpoint, incumbent, time) in snapshots.items()
                for decoder, mask in (("endpoint", endpoint), ("periodic", incumbent))]
    return rows, bias.cpu() if bias is not None else None


@torch.no_grad()
def evaluate(args):
    from eval_static_prior import write_rows

    data, graph_ids = cohort(args)
    run = Experiment(args, graph_ids)
    ranker = load_pave_v3(pave(args.ref))
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    warm = data[0]
    legal = SearchGraph(warm.num_nodes, warm.edge_index, seed=1017)
    warm = warm.to(run.device)
    for arm in run.arms:
        solver = run.policy(arm, graph_ids[0], args.offset, legal, 25)
        search(solver, warm, legal, 1017)
    sync(run.device)
    records = []
    for repeat in range(args.repeats):
        current = []
        for index, graph in enumerate(data, start=args.offset):
            gid = graph_ids[index-args.offset]
            edges = graph.edge_index
            graph = graph.to(run.device)
            rotation = index % len(run.arms)
            order = run.arms[rotation:] + run.arms[:rotation]
            if repeat % 2:
                order = order[::-1]
            for seed in search_seeds(index, args.seeds_per_graph):
                legal = SearchGraph(graph.num_nodes, edges, seed=seed)
                paired = []
                prepared = {}
                for budget in (args.budgets if args.protocol == "budget" else (None,)):
                    for position, arm in enumerate(order):
                        solver = run.policy(arm, gid, index, legal, max(args.steps))
                        rows, bias = solve_rows(solver, graph, edges, seed, ranker, args.steps, budget)
                        prepared[arm, rows[0]["protocol"] if budget else None] = bias
                        for row in rows:
                            paired.append(dict(graph_id=gid, search_seed=seed, repeat=repeat, order=position,
                                               arm=arm, **row))
                stats = measure_initial(run, graph, seed, prepared)
                baseline_sets = {(row["protocol"], row["decoder"]): set(row["selected_nodes"].split())
                                 for row in paired if row["arm"] == "b0"}
                for row in paired:
                    other = baseline_sets[row["protocol"], row["decoder"]]
                    selected = set(row["selected_nodes"].split())
                    row.update(**stats[row["arm"], row["protocol"] if args.protocol == "budget" else None],
                               final_jaccard=len(selected & other)/max(1, len(selected | other)))
                current.extend(paired)
            if (index-args.offset+1) % 10 == 0 or index-args.offset+1 == len(data):
                print(f"{args.protocol} repeat {repeat} {index-args.offset+1}/{len(data)}", flush=True)
        write_rows(args.output / f"repeat_{repeat}.csv", current)
        records.extend(current)
    source = provenance(args.ref)
    source.update(version="static_prior_v2", status="completed", backend=args.backend,
                  regime_note="curve time includes offline diagnostic decodes" if args.protocol == "curve"
                  else "deadline includes prepare and PAVE; mechanism replay is outside timing",
                  precision="fp32" if args.backend == "fp32-graph" else "fp16",
                  cuda_graph=args.backend == "fp32-graph", num_k=20, reserve_s=.05, decode_every=10,
                  graph_ids=graph_ids, search_seeds={gid: search_seeds(i, args.seeds_per_graph)
                                                   for i, gid in enumerate(graph_ids, args.offset)},
                  config=vars(args), feature_norm_sha256=run.norm_hashes,
                  oracle_sha256=run.target_hashes,
                  doses=run.doses,
                  random_rms=RANDOM_RMS, rng_protocol="same generator seed per graph and arm, separate fixed bias seed",
                  checkpoint_sha256={arm: digest(path) for arm, path in run.paths.items()},
                  alpha=run.alpha,
                  limits={arm: run.policies[arm].accelerator.unwrap_model(run.policies[arm].model).delta_limit
                          for arm in run.paths if arm != "b0"},
                  params={arm: sum(p.numel() for p in solver.model.parameters()) for arm, solver in run.policies.items()},
                  pave_params=sum(p.numel() for p in ranker.parameters()),
                  gpu=os.environ.get("CUDA_VISIBLE_DEVICES"), device=str(run.device), started_at=started,
                  finished_at=datetime.now(timezone.utc).isoformat())
    write_rows(args.output / "per_graph.csv", records)
    write_run(args.output, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    parser.add_argument("--protocol", choices=("curve", "budget"), required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=STEPS)
    parser.add_argument("--budgets", nargs="+", type=float, choices=BUDGETS, default=BUDGETS)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.offset < 0 or min(args.samples, args.repeats, args.seeds_per_graph, *args.steps) < 1:
        parser.error("Use a nonnegative offset and positive graph/seed/repeat/step counts")
    if len(set(args.budgets)) != len(args.budgets):
        parser.error("Use distinct budgets")
    if not math.isfinite(args.oracle_amplitude) or args.oracle_amplitude < 0:
        parser.error("Use a finite nonnegative oracle amplitude")
    evaluate(args)


if __name__ == "__main__":
    main()
