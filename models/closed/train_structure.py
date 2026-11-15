"""Train the structural compensation on frozen-B teacher trajectories.

Only ``compensation.*`` parameters move; the backbone stays frozen, so the run
isolates what the new channels contribute. Zeroing the compensation reproduces B
exactly, which gives a same-checkpoint "off" arm for free and a per-epoch effect
size (``comp_delta_rms``) comparable to the v1 masked control.
"""
import csv
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v2 import search, write_run
from main import build_parser, set_seed
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import load_pave_v3
from models import GCN
from solvers import RLNN
from v7_runs import backbone, digest, pave, provenance


class StructureRLNN(RLNN):
    def __init__(self, args):
        super().__init__(args)
        self.validation_offset = args.validation_offset
        self.valid_loader.dataset.files = self.valid_loader.dataset.files[self.validation_offset:]
        self.teacher_seed = 20261 + args.seed
        self.shuffle_seed = 30497 + args.seed
        self.train_loader.sampler.generator = torch.Generator().manual_seed(self.shuffle_seed)
        self.model.requires_grad_(False)
        self.compensation = [(name, parameter) for name, parameter in self.model.named_parameters()
                             if name.startswith("compensation.")]
        if not self.compensation:
            raise ValueError("This structure mode exposes no trainable compensation")
        for _, parameter in self.compensation:
            parameter.requires_grad_(True)
        self.trainable = [parameter for _, parameter in self.compensation]
        self.teacher = GCN(1, args.num_h, args.num_l, 1)
        self.teacher.load_state_dict(torch.load(args.warm_start, map_location="cpu", weights_only=True))
        self.teacher.to(self.accelerator.device).requires_grad_(False).eval()
        self.ranker = load_pave_v3(pave(args.reference_seed))
        self.optimizer = torch.optim.Adam(self.trainable, lr=args.lr)
        self.state_manifest = {}
        self.first_gradients = []
        self.optimizer.register_step_pre_hook(self.record_gradients)

    def record_gradients(self, *_):
        if len(self.first_gradients) < 2:
            gradients = [parameter.grad.detach().square().sum() for parameter in self.trainable
                         if parameter.grad is not None]
            self.first_gradients.append(dict(
                grad_norm=float(torch.stack(gradients).sum().sqrt()) if gradients else 0.,
                nonzero_tensors=sum(1 for parameter in self.trainable
                                    if parameter.grad is not None and parameter.grad.abs().sum() > 0)))

    @torch.no_grad()
    def generate_data(self, graph, A=None, normalized=None):
        device = self.accelerator.device
        sizes = [b - a for a, b in zip(graph.graph_ptr, graph.graph_ptr[1:])]
        initials, noise = [], []
        for index, size in zip(graph.graph_ids, sizes):
            seed = self.teacher_seed + 1_000_003 * self.training_epoch + index
            rng = torch.Generator(device=device).manual_seed(seed)
            initials.append(torch.randint(0, 2, (size, self.num_k), dtype=self.dtype, device=device, generator=rng))
            noise.append(torch.rand(self.num_tp, size, self.num_k, dtype=self.dtype, device=device, generator=rng))
        x, noise = torch.cat(initials), torch.cat(noise, dim=1)
        states = []
        for step in range(self.num_tp):
            states.append(x)
            probability = self.teacher(x.unsqueeze(-1), graph.edge_index, normalized.normalized).squeeze(-1).sigmoid()
            x = torch.where(noise[step] < probability, 1 - x, x)
        data = torch.cat(states, dim=1)
        if self.training_epoch == 0:
            for index, a, b in zip(graph.graph_ids, graph.graph_ptr, graph.graph_ptr[1:]):
                if index < 20:
                    self.state_manifest[index] = sha256(data[a:b].cpu().numpy().tobytes()).hexdigest()
        self.probe = (graph, A, normalized, data[:, :4])
        return data

    @torch.no_grad()
    def compensation_effect(self, graph, A, normalized, x):
        """Logit change caused by the compensation, with the same weights restored afterwards."""
        product = (A @ x) if self.structure_gain else None
        model = self.accelerator.unwrap_model(self.model)
        backup = [parameter.detach().clone() for parameter in self.trainable]
        full = model(x, graph.edge_index, A, normalized, product)
        for parameter in self.trainable:
            parameter.zero_()
        base = model(x, graph.edge_index, A, normalized, product)
        for parameter, value in zip(self.trainable, backup):
            parameter.copy_(value)
        delta = full - base
        return dict(comp_delta_rms=float(delta.square().mean().sqrt()),
                    comp_delta_prob_mean_abs=float(((base + delta).sigmoid() - base.sigmoid()).abs().mean()),
                    comp_delta_max=float(delta.abs().amax()))

    @torch.no_grad()
    def evaluate(self, mode, log_path=None):
        if mode != "valid":
            return super().evaluate(mode, log_path)
        self.model.eval()
        bare, pave_sizes, times, scores, steps = [], [], [], [], []
        for graphs in self.valid_loader:
            for graph in graphs:
                seed = self.validation_seed + self.validation_offset + graph.idx
                cpu = SearchGraph(graph.num_nodes, graph.edge_index.cpu(), seed=seed)
                tick = perf_counter()
                solution, count = search(self, graph, cpu, seed)
                decoded = perf_counter()
                improved = platform_search(cpu, solution, choose=self.ranker.choose, horizon=1).best
                assert cpu.feasible(improved)
                bare.append(solution.bit_count())
                pave_sizes.append(improved.bit_count())
                times.append(decoded - tick)
                scores.append(perf_counter() - decoded)
                steps.append(count)
        self.validation_metrics = dict(val_pave_size=mean(pave_sizes), val_post_s=mean(scores),
                                       val_steps=mean(steps))
        return -mean(bare), mean(times)

    @torch.no_grad()
    def print_log(self, logger):
        model = self.accelerator.unwrap_model(self.model)
        compensation = model.compensation
        if compensation.use_gate:
            logger["gate_tanh"] = [round(float(value), 4) for value in compensation.gate.tanh().tolist()]
        if compensation.use_gain:
            logger["gain_gate_tanh"] = float(compensation.gain_gate.tanh())
        if compensation.use_degree:
            norms = [float(layer.weight.square().sum().sqrt()) for layer in compensation.degree_proj]
            logger["degree_bias_norm"] = [round(value, 5) for value in norms]
        logger["trainable"] = sum(parameter.numel() for parameter in self.trainable)
        if hasattr(self, "probe"):
            graph, A, normalized, x = self.probe
            logger.update(**self.compensation_effect(graph, A, normalized, x))
        super().print_log(logger)


def main():
    parser = build_parser()
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--reference_seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--validation_offset", type=int, default=0)
    args = parser.parse_args()
    if args.structure_mode == "none" or args.resume or args.mixed_precision != "no" or args.graph_batch_size < 2:
        parser.error("Use a structure mode, FP32 and multi-graph training")
    if args.context_mode != "none" or args.frequency != "none":
        parser.error("The structure run cannot mix in another encoder")
    args.warm_start = str(backbone(args.reference_seed))
    args.do_train = True
    args.validation_seed = 1017
    args.save_dir = str(args.result_dir / "checkpoints")
    Path(args.save_dir).mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    solver = StructureRLNN(args)
    record = dict(**provenance(args.reference_seed), version="structure_v1", arm=args.structure_mode,
                  status="running", config=vars(args), started_at=datetime.now(timezone.utc).isoformat(),
                  teacher_state_seed=solver.teacher_seed, shuffle_seed=solver.shuffle_seed,
                  validation_graph_ids=[Path(file).stem for file in solver.valid_loader.dataset.files],
                  params=sum(p.numel() for p in solver.model.parameters()),
                  trainable_params=sum(parameter.numel() for parameter in solver.trainable),
                  trainable_names=[name for name, _ in solver.compensation])
    write_run(args.result_dir, record)
    started = perf_counter()
    try:
        solver.prepare()
        solver.train(args.save_dir)
        state = torch.load(Path(args.save_dir) / "latest_training.pt", map_location="cpu", weights_only=False)
        history = state["history"]
        with (args.result_dir / "training.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        record.update(status="completed", updates=history[-1]["Iteration"], epochs=state["epoch"],
                      best_validation_size=-state["best_eval"], first_gradients=solver.first_gradients,
                      teacher_state_sha256=solver.state_manifest,
                      checkpoint_sha256=digest(Path(args.save_dir) / "best.pt"),
                      peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
    except Exception:
        record.update(status="failed")
        raise
    finally:
        record.update(train_wall_s=perf_counter() - started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_run(args.result_dir, record)


if __name__ == "__main__":
    main()
