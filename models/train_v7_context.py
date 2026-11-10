"""Train one-shot action priors on independently seeded frozen-B trajectories."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v2 import TRAIN, dataset_at, search, write_run
from main import build_parser, set_seed
from mis_v1 import adjacency
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import load_pave_v3
from models import GCN
from probe_v7_context import normalizer
from solvers import RLNN
from v7_runs import backbone, digest, pave, provenance


class ContextRLNN(RLNN):
    def __init__(self, args):
        super().__init__(args)
        self.teacher_seed = 20261 + args.seed
        self.shuffle_seed = 30497 + args.seed
        self.train_loader.sampler.generator = torch.Generator().manual_seed(self.shuffle_seed)
        self.model.requires_grad_(False)
        self.model.adapter.requires_grad_(True)
        self.teacher = GCN(1, args.num_h, args.num_l, 1)
        self.teacher.load_state_dict(torch.load(args.warm_start, map_location="cpu", weights_only=True))
        self.teacher.to(self.accelerator.device).requires_grad_(False).eval()
        self.ranker = load_pave_v3(pave(args.reference_seed))
        self.state_manifest = {}
        self.first_gradients = []
        alpha = self.model.adapter.alpha
        other = [p for p in self.model.adapter.parameters() if p is not alpha]
        self.optimizer = torch.optim.Adam([dict(params=other), dict(params=[alpha], lr=args.alpha_lr)], lr=args.lr)
        self.optimizer.register_step_pre_hook(self.record_gradients)

    def record_gradients(self, *_):
        if len(self.first_gradients) < 2:
            self.first_gradients.append(self.gradient_metrics())

    def gradient_metrics(self):
        adapter = self.accelerator.unwrap_model(self.model).adapter
        encoder = getattr(adapter, "sequence", getattr(adapter, "context", adapter.out_proj))
        gradients = [p.grad.detach().square().sum() for p in encoder.parameters() if p.grad is not None]
        return dict(alpha_grad=float(adapter.alpha.grad.abs()) if adapter.alpha.grad is not None else None,
                    encoder_grad_norm=float(torch.stack(gradients).sum().sqrt()) if gradients else 0.)

    @torch.no_grad()
    def generate_data(self, graph, A=None, normalized=None):
        device = self.accelerator.device
        sizes = [b-a for a, b in zip(graph.graph_ptr, graph.graph_ptr[1:])]
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
            x = torch.where(noise[step] < probability, 1-x, x)
        data = torch.cat(states, dim=1)
        if self.training_epoch == 0:
            for index, a, b in zip(graph.graph_ids, graph.graph_ptr, graph.graph_ptr[1:]):
                if index < 20:
                    self.state_manifest[index] = sha256(data[a:b].cpu().numpy().tobytes()).hexdigest()
        self.probe = (graph, A, normalized, data[:, :4])
        return data

    @torch.no_grad()
    def evaluate(self, mode, log_path=None):
        if mode != "valid":
            return super().evaluate(mode, log_path)
        self.model.eval()
        bare, pave_sizes, times, scores, steps = [], [], [], [], []
        for graphs in self.valid_loader:
            for graph in graphs:
                seed = self.validation_seed + graph.idx
                cpu = SearchGraph(graph.num_nodes, graph.edge_index.cpu(), seed=seed)
                tick = perf_counter()
                solution, count = search(self, graph, cpu, seed)
                decoded = perf_counter()
                improved = platform_search(cpu, solution, choose=self.ranker.choose, horizon=1).best
                assert cpu.feasible(improved)
                bare.append(solution.bit_count())
                pave_sizes.append(improved.bit_count())
                times.append(decoded-tick)
                scores.append(perf_counter()-decoded)
                steps.append(count)
        self.validation_metrics = dict(val_pave_size=mean(pave_sizes), val_post_s=mean(scores), val_steps=mean(steps))
        return -mean(bare), mean(times)

    @torch.no_grad()
    def print_log(self, logger):
        model = self.accelerator.unwrap_model(self.model)
        logger["alpha"] = float(model.adapter.alpha)
        if hasattr(self, "probe"):
            graph, _, topology, x = self.probe
            bias = model.adapter.precompute(topology)
            scaled = model.adapter.scaled(bias, model.delta_limit)
            delta = torch.where(x.bool(), scaled[:, 1:2], scaled[:, 0:1])
            base = self.teacher(x.unsqueeze(-1), graph.edge_index, topology.normalized).squeeze(-1)
            logger.update(static_bias_std=float(bias.std()), add_mean=float(bias[:, 0].mean()),
                          drop_mean=float(bias[:, 1].mean()), delta_logit_rms=float(delta.square().mean().sqrt()),
                          delta_prob_mean_abs=float(((base+delta).sigmoid()-base.sigmoid()).abs().mean()),
                          **self.gradient_metrics(), **self.validation_metrics)
        super().print_log(logger)


def main():
    parser = build_parser()
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--reference_seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--alpha_lr", type=float, default=.001)
    args = parser.parse_args()
    if args.context_mode == "none" or args.resume or args.mixed_precision != "no" or args.graph_batch_size < 2:
        parser.error("Use a v7 context, FP32 and multi-graph training")
    args.warm_start = str(backbone(args.reference_seed))
    args.do_train = True
    args.validation_seed = 1017
    args.save_dir = str(args.result_dir / "checkpoints")
    run = args.result_dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    solver = ContextRLNN(args)
    adapter = solver.model.adapter
    feature_mean, feature_std = normalizer(dataset_at(TRAIN, torch.float32, 1000))
    adapter.feature_mean.copy_(torch.from_numpy(feature_mean))
    adapter.feature_std.copy_(torch.from_numpy(feature_std))
    normalization = dict(mean=feature_mean.tolist(), std=feature_std.tolist())
    record = dict(**provenance(args.reference_seed), version="v7_context", arm=args.context_mode,
                  status="running", config=vars(args), started_at=datetime.now(timezone.utc).isoformat(),
                  teacher_state_seed=solver.teacher_seed, shuffle_seed=solver.shuffle_seed,
                  feature_normalization=normalization,
                  feature_norm_sha256=sha256(feature_mean.tobytes()+feature_std.tobytes()).hexdigest(),
                  params=sum(p.numel() for p in solver.model.parameters()),
                  trainable_params=sum(p.numel() for p in solver.model.parameters() if p.requires_grad))
    write_run(run, record)
    started = perf_counter()
    try:
        solver.prepare()
        solver.train(args.save_dir)
        state = torch.load(Path(args.save_dir) / "latest_training.pt", map_location="cpu", weights_only=False)
        history = state["history"]
        with (run / "training.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        record.update(status="completed", updates=history[-1]["Iteration"], epochs=state["epoch"],
                      best_validation_size=-state["best_eval"], first_gradients=solver.first_gradients,
                      teacher_state_sha256=solver.state_manifest,
                      adapter_checkpoint_sha256=digest(Path(args.save_dir) / "best.pt"),
                      peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
    except Exception:
        record.update(status="failed")
        raise
    finally:
        record.update(train_wall_s=perf_counter()-started, finished_at=datetime.now(timezone.utc).isoformat())
        write_run(run, record)


if __name__ == "__main__":
    main()
