"""Train one-shot action priors on independently seeded frozen-B trajectories."""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import atanh, isfinite
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from eval_v2 import TRAIN, dataset_at, search, write_run
from main import build_parser, set_seed
from mis_static_prior import STATIC_MODES, bias_stats, cheap_features, graph_center, oracle_bias
from mis_v1 import adjacency
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import load_pave_v3
from models import GCN
from solvers import RLNN
from v7_runs import backbone, digest, pave, provenance
from v7_structure import offline_structure


class ContextRLNN(RLNN):
    def __init__(self, args):
        super().__init__(args)
        self.validation_offset = args.validation_offset
        if not 0 <= self.validation_offset < len(self.valid_loader.dataset):
            raise ValueError("Validation offset must leave at least one graph")
        self.valid_loader.dataset.files = self.valid_loader.dataset.files[self.validation_offset:]
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
        self.training_manifest = {}
        self.student_columns = {}
        self.first_gradients = []
        self.static_v2 = args.static_v2
        self.cuda_graph_search = self.static_v2 and self.accelerator.device.type == "cuda"
        self.student_particles = round(args.student_mix * self.num_k)
        self.student_mix_every = args.student_mix_every
        self.generated_batches = 0
        self.validation_steps = (100, 400, 800) if self.static_v2 else (self.num_t,)
        self.oracle_labels = None
        adapter = self.model.adapter
        if args.oracle_train_targets:
            if Path(args.train_path).resolve() != TRAIN.resolve():
                raise ValueError("Oracle supervision requires the locked training split")
            targets = json.loads(args.oracle_train_targets.read_text(encoding="utf-8"))
            ids = [Path(path).stem for path in self.train_loader.dataset.files]
            missing = [gid for gid in ids if gid not in targets]
            if missing:
                raise ValueError(f"{args.oracle_train_targets} covers {len(ids)-len(missing)}/{len(ids)} training graphs; "
                                 f"first missing target is {missing[0]}. Regenerate with --train-targets --samples {len(ids)}.")
            self.oracle_labels = [targets[gid] for gid in ids]
            self.oracle_weight = args.oracle_weight
            self.oracle_loss_sums = torch.zeros(2, device=self.accelerator.device)
            self.oracle_updates = 0
            with torch.no_grad():
                adapter.alpha.fill_(atanh(.5))
            adapter.alpha.requires_grad_(False)
        gates = [p for name, p in adapter.named_parameters() if p.requires_grad and name in ("alpha", "constants")]
        other = [p for p in adapter.parameters() if p.requires_grad and all(p is not gate for gate in gates)]
        self.optimizer = torch.optim.Adam([dict(params=other), dict(params=gates, lr=args.alpha_lr)], lr=args.lr)
        self.optimizer.register_step_pre_hook(self.record_gradients)

    def record_gradients(self, *_):
        if len(self.first_gradients) < 2:
            self.first_gradients.append(self.gradient_metrics())

    def gradient_metrics(self):
        adapter = self.accelerator.unwrap_model(self.model).adapter
        encoder = next((getattr(adapter, name) for name in ("encoder", "sequence", "context", "out_proj")
                        if getattr(adapter, name, None) is not None), None)
        gradients = [p.grad.detach().square().sum() for p in encoder.parameters() if p.grad is not None] if encoder is not None else []
        alpha = getattr(adapter, "alpha", None)
        result = dict(alpha_grad=float(alpha.grad.abs()) if alpha is not None and alpha.grad is not None else None,
                      encoder_grad_norm=float(torch.stack(gradients).sum().sqrt()) if gradients else 0.)
        if hasattr(adapter, "constants"):
            result["constants_grad_norm"] = float(adapter.constants.grad.norm()) if adapter.constants.grad is not None else None
        return result

    def graph_batch_loss(self, graph, adjacency, topology, x, prob, *, terms):
        result = super().graph_batch_loss(graph, adjacency, topology, x, prob, terms=terms)
        if self.oracle_labels is None:
            return result
        adapter = self.accelerator.unwrap_model(self.model).adapter
        direction = adapter.direction(topology, normalize=adapter.peak_normalize)
        labels = torch.cat([oracle_bias(end-start, self.oracle_labels[index], 1., direction.device)
                            for index, start, end in zip(graph.graph_ids, graph.graph_ptr, graph.graph_ptr[1:])])
        alignment = (direction - graph_center(labels, graph.graph_ptr, normalize=adapter.peak_normalize)).square().mean()
        erdoes = result[0] if terms else result
        self.oracle_loss_sums += torch.stack((erdoes.detach(), alignment.detach()))
        self.oracle_updates += 1
        objective = erdoes + self.oracle_weight * alignment
        return (objective, result[1]) if terms else objective

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
        initial, noise = torch.cat(initials), torch.cat(noise, dim=1)
        x = initial
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
        columns = None
        if self.student_particles and self.generated_batches % self.student_mix_every == 0:
            model = self.accelerator.unwrap_model(self.model)
            student_topology = normalized._replace(cached_action_bias=model.adapter.action_bias(normalized, model.delta_limit))
            k = self.student_particles
            # Rotate the replaced particle columns across batches while keeping their random streams.
            columns = (torch.arange(k, device=device) + self.generated_batches // self.student_mix_every) % self.num_k
            student = initial[:, columns]
            for step in range(self.num_tp):
                states[step] = states[step].clone()
                states[step][:, columns] = student
                probability = self.flip_logits(student, graph, A, student_topology).sigmoid()
                student = torch.where(noise[step, :, columns] < probability, 1-student, student)
            data = torch.cat(states, dim=1)
        if self.static_v2 and self.training_epoch == 0:
            replaced = columns.tolist() if columns is not None else []
            for index, a, b in zip(graph.graph_ids, graph.graph_ptr, graph.graph_ptr[1:]):
                if index < 20:
                    self.training_manifest[index] = (self.state_manifest[index] if not replaced else
                                                     sha256(data[a:b].cpu().numpy().tobytes()).hexdigest())
                    self.student_columns[index] = replaced
        self.generated_batches += 1
        return data

    @torch.no_grad()
    def evaluate(self, mode, log_path=None):
        if mode != "valid":
            return super().evaluate(mode, log_path)
        self.model.eval()
        bare = {step: [] for step in self.validation_steps}
        pave_sizes = {step: [] for step in self.validation_steps}
        times, scores = [], []
        final_step = max(self.validation_steps)
        for graphs in self.valid_loader:
            for graph in graphs:
                seed = self.validation_seed + self.validation_offset + graph.idx
                cpu = SearchGraph(graph.num_nodes, graph.edge_index.cpu(), seed=seed)
                tick = perf_counter()
                endpoints = {}

                def record(step, selected, _, periodic):
                    if step in bare:
                        endpoints[step] = selected

                original_steps = self.num_t
                self.num_t = final_step
                try:
                    search(self, graph, cpu, seed, checkpoints=self.validation_steps, on_decode=record)
                finally:
                    self.num_t = original_steps
                decoded = perf_counter()
                for step, solution in endpoints.items():
                    improved = platform_search(cpu, solution, choose=self.ranker.choose, horizon=1).best
                    assert cpu.feasible(improved)
                    bare[step].append(solution.bit_count())
                    pave_sizes[step].append(improved.bit_count())
                times.append(decoded-tick)
                scores.append(perf_counter()-decoded)
        first = self.validation_steps[0]
        self.validation_metrics = dict(val_pave_size=mean(pave_sizes[first]), val_post_s=mean(scores), val_steps=final_step)
        if self.static_v2:
            for step in self.validation_steps:
                self.validation_metrics.update({f"val_a3_t{step}": mean(bare[step]),
                                                f"val_pave_t{step}": mean(pave_sizes[step])})
        return -mean(bare[first]), mean(times)

    @torch.no_grad()
    def paired_probe(self, graph, adjacency, topology, x, bias):
        rng = torch.Generator(device=x.device).manual_seed(self.teacher_seed + self.training_epoch)
        teacher, student = x, x
        topology = topology._replace(cached_action_bias=bias)
        result = {}
        for step in range(1, 101):
            p = self.teacher(teacher.unsqueeze(-1), graph.edge_index, topology.normalized).squeeze(-1).sigmoid()
            q = self.flip_logits(student, graph, adjacency, topology).sigmoid()
            uniform = torch.rand(teacher.shape, generator=rng, device=x.device, dtype=x.dtype)
            teacher = torch.where(uniform < p, 1-teacher, teacher)
            student = torch.where(uniform < q, 1-student, student)
            if step in (25, 50, 100):
                result[f"divergence_t{step}"] = float((teacher != student).float().mean())
        return result

    @torch.no_grad()
    def print_log(self, logger):
        model = self.accelerator.unwrap_model(self.model)
        adapter = model.adapter
        logger["alpha"] = float(adapter.alpha) if hasattr(adapter, "alpha") else None
        if self.oracle_labels is not None and self.oracle_updates:
            erdoes, alignment = (self.oracle_loss_sums / self.oracle_updates).tolist()
            logger.update(train_erdoes_loss=erdoes, train_oracle_mse=alignment,
                          train_weighted_oracle_loss=self.oracle_weight * alignment)
            self.oracle_loss_sums.zero_()
            self.oracle_updates = 0
        if hasattr(self, "probe"):
            graph, adj, topology, x = self.probe
            if self.context_mode in STATIC_MODES:
                bias = scaled = adapter.action_bias(topology, model.delta_limit)
                logger["c_add"], logger["c_drop"] = adapter.constants.tolist()
            else:
                bias = adapter.precompute(topology)
                scaled = adapter.scaled(bias, model.delta_limit)
            delta = torch.where(x.bool(), scaled[:, 1:2], scaled[:, 0:1])
            base = self.teacher(x.unsqueeze(-1), graph.edge_index, topology.normalized).squeeze(-1)
            logger.update(static_bias_std=float(bias.std()), add_mean=float(bias[:, 0].mean()),
                          drop_mean=float(bias[:, 1].mean()), delta_logit_rms=float(delta.square().mean().sqrt()),
                          delta_prob_mean_abs=float(((base+delta).sigmoid()-base.sigmoid()).abs().mean()),
                          **self.gradient_metrics(), **self.validation_metrics)
            if self.static_v2:
                p, q = base.sigmoid(), (base + delta).sigmoid()
                logger.update(bias_stats(bias))
                logger.update(boundary_fraction=float(((q > .01) & (q < .99)).float().mean()),
                              flip_rate=float(q.sum()/((len(topology.ptr)-1)*x.shape[1])),
                              base_flip_rate=float(p.sum()/((len(topology.ptr)-1)*x.shape[1])),
                              **self.paired_probe(graph, adj, topology, x, bias))
        super().print_log(logger)


def main():
    parser = build_parser()
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--reference_seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--alpha_lr", type=float, default=.001)
    parser.add_argument("--validation_offset", type=int, default=0)
    parser.add_argument("--static-v2", dest="static_v2", action="store_true")
    parser.add_argument("--student_mix", type=float, default=0.)
    parser.add_argument("--student_mix_every", type=int, default=1, help="Refresh every N graph batches")
    parser.add_argument("--oracle_train_targets", type=Path)
    parser.add_argument("--oracle_weight", "--oracle-weight", type=float, default=.03)
    args = parser.parse_args()
    if args.context_mode == "none" or args.resume or args.mixed_precision != "no" or args.graph_batch_size < 2:
        parser.error("Use a v7 context, FP32 and multi-graph training")
    if not 0 <= args.student_mix <= 1 or args.student_mix_every < 1:
        parser.error("Use student_mix in [0,1] and a positive refresh interval")
    if args.context_mode in ("static_random", "static_oracle"):
        parser.error("Random and oracle modes are inference interventions")
    if (args.static_v2 or args.student_mix or args.oracle_train_targets) and args.context_mode not in ("static_centered", "static_constants"):
        parser.error("v2 training requires a static adapter")
    if args.oracle_train_targets and args.context_mode != "static_centered":
        parser.error("Oracle direction training requires the centered adapter")
    if args.oracle_train_targets and not (isfinite(args.oracle_weight) and args.oracle_weight > 0):
        parser.error("Oracle direction training requires a positive finite --oracle_weight")
    if args.oracle_train_targets:
        args.static_peak_normalize = True
    args.warm_start = str(backbone(args.reference_seed))
    args.do_train = True
    args.validation_seed = 1017
    args.save_dir = str(args.result_dir / "checkpoints")
    run = args.result_dir
    Path(args.save_dir).mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    solver = ContextRLNN(args)
    adapter = solver.model.adapter
    normalization = norm_digest = None
    if args.context_mode != "static_constants":
        if args.context_mode == "static_centered":
            if args.static_features == 5:
                values = torch.cat([cheap_features(adjacency(graph), (0, graph.num_nodes))
                                    for graph in solver.train_loader.dataset])
            else:
                order = adapter.encoder.order if adapter.encoder is not None else None
                values = torch.cat([torch.tensor(offline_structure(
                    graph.num_nodes, graph.edge_index.numpy().tobytes(), adapter.order_seed + graph.idx,
                    order, adapter.views)[0]) for graph in solver.train_loader.dataset])
            feature_mean, feature_std = values.mean(dim=0), values.std(dim=0, correction=0).clamp_min(1e-4)
        else:
            from probe_v7_context import normalizer

            feature_mean, feature_std = (torch.from_numpy(x) for x in normalizer(dataset_at(TRAIN, torch.float32, 1000)))
        adapter.feature_mean.copy_(feature_mean)
        adapter.feature_std.copy_(feature_std)
        normalization = dict(mean=feature_mean.tolist(), std=feature_std.tolist())
        norm_digest = sha256(feature_mean.numpy().tobytes()+feature_std.numpy().tobytes()).hexdigest()
    record = dict(**provenance(args.reference_seed), version="v7_context", arm=args.context_mode,
                  status="running", config=vars(args), started_at=datetime.now(timezone.utc).isoformat(),
                  teacher_state_seed=solver.teacher_seed, shuffle_seed=solver.shuffle_seed,
                  feature_normalization=normalization,
                  feature_norm_sha256=norm_digest,
                  validation_graph_ids=[Path(file).stem for file in solver.valid_loader.dataset.files],
                  params=sum(p.numel() for p in solver.model.parameters()),
                  trainable_params=sum(p.numel() for p in solver.model.parameters() if p.requires_grad))
    if args.context_mode in STATIC_MODES:
        record["version"] = "static_prior_v2" if args.static_v2 else "static_prior_v1"
        record["static_feature_schema"] = adapter.feature_schema if normalization else ()
    if args.oracle_train_targets:
        record["oracle_train_targets_sha256"] = digest(args.oracle_train_targets)
        record["oracle_fixed_gate"] = float(adapter.alpha.tanh())
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
        if args.static_v2:
            record.update(training_state_sha256=solver.training_manifest, student_columns=solver.student_columns)
    except Exception:
        record.update(status="failed")
        raise
    finally:
        record.update(train_wall_s=perf_counter()-started, finished_at=datetime.now(timezone.utc).isoformat())
        write_run(run, record)


if __name__ == "__main__":
    main()
