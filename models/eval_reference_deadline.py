"""Evaluate MIS reference solvers with a common per-graph deadline."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from eval_v2 import TEST_CACHE, dataset_at
from timed_mis_archive import CLOCK, TimedArchive


SEEDS = (1017, 1019, 1021)
AUTHOR_CHECKPOINT = Path("models/data/author_rld4co/checkpoints/erdoes/rlnn_rb_small_mis")
AUTHOR_WEIGHT_URL = ("https://huggingface.co/shengyuf/RLNN/resolve/"
                     "0b96a439a11ac90dc69512720084d08784f02bf4/erdoes/rlnn_rb_small_mis/best.pt")


class TimedRLSA(TimedArchive):
    def __init__(self, dataset):
        from main import build_parser
        from solvers import RLSA

        args = build_parser().parse_args([])
        args.test_path = str(dataset)
        args.method = "RLSA"
        args.mixed_precision = "fp16"
        args.num_k, args.num_t, args.num_d, args.tau0 = 200, 300, 5, 0.01
        self.solver = RLSA(args)
        self.solver.prepare()
        super().__init__(self.solver.accelerator.device, args.num_t)


class TimedAuthorRLNN(TimedArchive):
    def __init__(self, dataset, checkpoint):
        from eval_v2 import load_policy
        from main import build_parser

        args = build_parser().parse_args([])
        args.test_path = str(dataset)
        args.save_dir = str(checkpoint)
        args.method, args.loss, args.mixed_precision = "RLNN", "erdoes", "fp16"
        args.num_k, args.num_t, args.num_l, args.num_h = 20, 100, 5, 128
        self.solver = load_policy(args)
        super().__init__(self.solver.accelerator.device, args.num_t)

    def initialize(self, graph, generator):
        from mis_v1 import adjacency

        adj = adjacency(graph)
        topology = self.solver.policy_topology(graph, adj)
        samples = self.solver.sample_states(graph, adj, generator=generator, normalized=topology)
        best = next(samples)
        energy = self.solver.problem.energy_func(adj, graph.b, best)[0]
        return adj, (samples, best, energy)

    def advance(self, graph, adj, state, steps, generator):
        samples, best, best_energy = state
        for _ in range(10):
            x = next(samples)
            energy = self.solver.problem.energy_func(adj, graph.b, x)[0]
            better = energy < best_energy
            best = torch.where(better.unsqueeze(0), x, best)
            best_energy = torch.where(better, energy, best_energy)
        return samples, best, best_energy

    def decode(self, graph, adj, state):
        from mis_v1 import best_decoded

        mask = best_decoded(self.solver.problem, state[1], adj, graph.b)
        return mask.nonzero().flatten().cpu().numpy()


class TimedReduMIS:
    def __init__(self, binary, directory, internal_limit_ms=180):
        self.binary = binary.resolve()
        self.directory = Path(directory)
        self.internal_limit_ms = internal_limit_ms
        self.steps = None
        self.info = {}

    def prepare(self, graph):
        from eval_kamis_anytime import neighbours_of

        path = self.directory / "input.graph"
        neighbours = neighbours_of(graph)
        lines = [f"{graph.num_nodes} {graph.edge_index.size(1) // 2}"]
        lines += [" ".join(str(node + 1) for node in sorted(row)) for row in neighbours]
        path.write_text("\n".join(lines) + "\n", encoding="ascii")
        return path, graph.num_nodes

    def warmup(self, graph):
        pass

    def solve(self, graph, seed, tick, clock):
        path, n = graph
        target = self.directory / "solution"
        target.unlink(missing_ok=True)
        command = [str(self.binary), str(path), f"--output={target}", f"--seed={seed}",
                   f"--time_limit={self.internal_limit_ms / 1000:g}", "--console_log"]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            stdout, _ = process.communicate(timeout=max(0, clock - (perf_counter() - tick)))
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _ = process.communicate()
            self.info = {"status": "deadline_timeout", "log": stdout.decode(errors="replace")}
            return
        self.info = {"status": "returned", "log": stdout.decode(errors="replace")}
        if process.returncode or not target.is_file():
            raise RuntimeError(f"ReduMIS exit {process.returncode}: {self.info['log']}")
        mask = np.fromstring(target.read_text(encoding="ascii"), sep=" ", dtype=int)
        if mask.size != n or np.any((mask != 0) & (mask != 1)):
            raise ValueError("ReduMIS returned an invalid node mask")
        nodes = np.flatnonzero(mask)
        ready = perf_counter() - tick
        if ready <= clock:
            yield ready, nodes


def validate(nodes, edges, count):
    mask = np.zeros(count, dtype=bool)
    mask[nodes] = True
    if (mask[edges[0]] & mask[edges[1]]).any():
        raise ValueError("solver emitted a non-independent set")


def evaluate(args, graphs, solver):
    args.output.mkdir(parents=True, exist_ok=False)
    solver.warmup(solver.prepare(graphs[0]))
    meta = dict(method=args.method, graphs=len(graphs), seeds=args.seeds, budget_ms=CLOCK * 1000,
                dataset=str(args.dataset), host=platform.node(), platform=platform.platform(),
                cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                solver_config=args.config if hasattr(args, "config") else solver.protocol(), torch=torch.__version__,
                timing=getattr(solver, "timing", "Prepared input outside; initialization, synchronized search, and host solution delivery inside."))
    (args.output / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    fields = ("graph_id", "seed", "repeat", "budget", "final_size", "wall_ms", "final_ready_ms",
              "first_ready_ms", "steps", "deadline_overrun", "has_solution")
    with (args.output / "per_graph.csv").open("w", newline="", encoding="utf-8") as values, \
         (args.output / "solutions.jsonl").open("w", encoding="utf-8") as solutions, \
         (args.output / "solver_log.jsonl").open("w", encoding="utf-8") as logs:
        writer = csv.DictWriter(values, fieldnames=fields)
        writer.writeheader()
        for index, graph in enumerate(graphs):
            prepared = solver.prepare(graph)
            edges = graph.edge_index.numpy()
            for seed in args.seeds:
                tick = perf_counter()
                events = list(solver.solve(prepared, seed, tick, CLOCK))
                wall = perf_counter() - tick
                key = dict(graph_id=f"idx_{index}", seed=seed)
                updates = []
                for ready, nodes in events:
                    validate(nodes, edges, graph.num_nodes)
                    updates.append(dict(ready_ms=ready * 1000, nodes=nodes.tolist()))
                last = updates[-1] if updates else None
                writer.writerow(dict(**key, repeat=0, budget=CLOCK,
                                     final_size=len(last['nodes']) if last else 0, wall_ms=wall * 1000,
                                     final_ready_ms=last['ready_ms'] if last else None,
                                     first_ready_ms=updates[0]['ready_ms'] if last else None,
                                     steps=solver.steps, deadline_overrun=int(wall > CLOCK), has_solution=int(bool(last))))
                solutions.write(json.dumps(dict(**key, updates=updates), separators=(",", ":")) + "\n")
                if hasattr(solver, "info"):
                    logs.write(json.dumps(dict(**key, **solver.info)) + "\n")
            if (index + 1) % 20 == 0 or index + 1 == len(graphs):
                values.flush()
                solutions.flush()
                logs.flush()
                print(f"{args.method}: {index + 1}/{len(graphs)} graphs", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("isco", "rlsa", "kamis", "author_rlnn", "fld_eg", "diffuco", "ltft",
                                          "difusco", "dgl_treesearch", "dimes", "intel", "ppo_lwd"))
    parser.add_argument("--dataset", type=Path, default=TEST_CACHE)
    parser.add_argument("--graphs", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kamis", type=Path, default=Path("tmp/kamis_build/redumis"))
    parser.add_argument("--kamis-internal-ms", type=int, default=180)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ltft-config", type=Path)
    parser.add_argument("--ltft-device", default="cuda:0")
    parser.add_argument("--isco-max-steps", type=int, choices=(300,), help="match the paper's 300-step cap")
    args = parser.parse_args()
    dtype = torch.float16 if args.method in ("rlsa", "author_rlnn") else torch.float32
    graphs = dataset_at(args.dataset, dtype, args.graphs, official=True)
    if args.method == "isco":
        from official_isco_deadline import OfficialISCO, jax

        if jax.default_backend() != "gpu":
            raise RuntimeError(f"DISCS requires GPU, got {jax.devices()}")
        args.config = dict(revision="aa8c12281a790fb36e98b32cc52279041b107f4a", jax=jax.__version__,
                           accelerator=str(jax.devices()[0]), sampler="DISCS adaptive SQRT PAFS",
                           cooling_horizon=300, max_steps=args.isco_max_steps,
                           particles=200, block_steps=10, penalty=1.001,
                           temperature="DISCS ertest exponential schedule with RB horizon 300",
                           candidate="best feasible DISCS sample")
        evaluate(args, graphs, OfficialISCO(graphs, max_steps=args.isco_max_steps))
    elif args.method == "rlsa":
        args.config = dict(horizon=300, particles=200, block_steps=10, radius=5, tau0=0.01,
                           precision="fp16", decoding="A.3 midpoint and terminal over all particles")
        evaluate(args, graphs, TimedRLSA(args.dataset))
    elif args.method == "author_rlnn":
        checkpoint = args.checkpoint or AUTHOR_CHECKPOINT
        official_dir = AUTHOR_CHECKPOINT.parent
        if checkpoint.parent == official_dir:
            checkpoint_source = AUTHOR_WEIGHT_URL.replace(AUTHOR_CHECKPOINT.name, checkpoint.name)
        else:
            checkpoint_source = str(checkpoint)
        args.config = dict(checkpoint=str(checkpoint), checkpoint_source=checkpoint_source,
                           precision="fp16", particles=20, horizon=100, block_steps=10, compile=False,
                           decoding="A.3 midpoint and terminal over all particles")
        evaluate(args, graphs, TimedAuthorRLNN(args.dataset, checkpoint))
    elif args.method == "fld_eg":
        from fld_eg_deadline import TimedFLDEG

        args.config = TimedFLDEG.protocol()
        evaluate(args, graphs, TimedFLDEG())
    elif args.method == "diffuco":
        from official_diffuco_deadline import OfficialDiffUCO

        if args.checkpoint is None:
            parser.error("DiffUCO requires --checkpoint")
        solver = OfficialDiffUCO(graphs, args.checkpoint)
        args.config = solver.protocol()
        evaluate(args, graphs, solver)
    elif args.method == "ltft":
        from official_ltft_deadline import OfficialLTFT

        if args.checkpoint is None or args.ltft_config is None:
            parser.error("LTFT requires --checkpoint and --ltft-config")
        solver = OfficialLTFT(args.checkpoint, args.ltft_config, args.ltft_device)
        args.config = solver.protocol()
        evaluate(args, graphs, solver)
    elif args.method == "difusco":
        from official_difusco_deadline import OfficialDIFUSCO

        if args.checkpoint is None:
            parser.error("DIFUSCO requires --checkpoint")
        evaluate(args, graphs, OfficialDIFUSCO(graphs, args.checkpoint))
    elif args.method == "dgl_treesearch":
        from official_dgl_treesearch_deadline import OfficialDGLTreeSearch

        if args.checkpoint is None:
            parser.error("DGL-TreeSearch requires --checkpoint")
        evaluate(args, graphs, OfficialDGLTreeSearch(args.checkpoint))
    elif args.method == "dimes":
        from dimes_deadline import DimesSearch

        if args.checkpoint is None:
            parser.error("DIMES requires the exported MIS heatmap --checkpoint")
        evaluate(args, graphs, DimesSearch(args.checkpoint))
    elif args.method == "intel":
        from intel_treesearch_deadline import IntelTreeSearch

        if args.checkpoint is None:
            parser.error("INTEL requires the exported --checkpoint")
        evaluate(args, graphs, IntelTreeSearch(args.checkpoint))
    elif args.method == "ppo_lwd":
        from official_ppo_lwd_deadline import OfficialPPOLwD

        if args.checkpoint is None:
            parser.error("PPO-LwD requires the trained actor --checkpoint")
        evaluate(args, graphs, OfficialPPOLwD(args.checkpoint))
    else:
        if not 0 < args.kamis_internal_ms <= CLOCK * 1000:
            parser.error("--kamis-internal-ms must be positive and fit the 200 ms host deadline")
        args.config = dict(binary=str(args.kamis), internal_limit_ms=args.kamis_internal_ms,
                           outer_limit_ms=200, omp_threads=os.getenv("OMP_NUM_THREADS"))
        with TemporaryDirectory() as directory:
            evaluate(args, graphs, TimedReduMIS(args.kamis, directory, args.kamis_internal_ms))


if __name__ == "__main__":
    main()
