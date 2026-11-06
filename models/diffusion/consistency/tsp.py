import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from diffusion.consistency.meta import MetaConsistency
from diffusion.utils.tsp_utils import TSPEvaluator, batched_two_opt_torch, merge_sub_tours

import time


class TSPConsistency(MetaConsistency):
    def __init__(
        self,
        _args,
        sigma_max=1000,
        sigma_min=0,
        weight_schedule="uniform",
        boundary_func="truncate",
    ):
        super(TSPConsistency, self).__init__(
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            weight_schedule=weight_schedule,
            boundary_func=boundary_func,
        )

    def consistency_losses(self, model, batch):
        edge_index = None
        if model.sparse:
            _, graph_data, point_indicator, _, _ = batch
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
            t = torch.randint(1, model.diffusion.T + 1, [batch_size], device=points.device)
        else:
            _, points, adj_matrix, _ = batch
            batch_size = points.shape[0]
            t = torch.randint(1, model.diffusion.T + 1, [batch_size], device=points.device)

        x0 = F.one_hot(adj_matrix.long(), num_classes=2).float()
        if model.sparse:
            x0 = x0.unsqueeze(1)

        if model.args.consistency:
            t2 = (model.args.alpha * t).int()
            x_t, x_t2 = model.diffusion.consistency_sample(
                x0, t.cpu().numpy(), t2.cpu().numpy()
            )
        else:
            x_t, _ = model.diffusion.sample(x0, t)
            t2 = None
            x_t2 = None

        if model.sparse:
            t = t.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
            x_t = x_t.reshape(-1)
            if model.args.consistency:
                t2 = t2.reshape(-1, 1).repeat(1, adj_matrix.shape[1]).reshape(-1)
                x_t2 = x_t2.reshape(-1)
            adj_matrix = adj_matrix.reshape(-1)
            points = points.reshape(-1, 2)
            edge_index = edge_index.float().to(adj_matrix.device).reshape(2, -1)

        _, denoise = self.denoise(model, points, x_t, t, edge_index, x0)
        adj_matrix = adj_matrix.long()
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(denoise, adj_matrix)
        if model.args.consistency:
            _, denoise2 = self.denoise(
                model, points, x_t2, t2, edge_index, x0
            )
            loss = loss + loss_func(denoise2, adj_matrix)

        return loss

    def denoise(self, model, points, x_t, t, edge_index, x0):
        x_t = x_t * 2 - 1
        x_t = x_t * (1.0 + 0.05 * torch.rand_like(x_t))
        model_output = model(points, x_t, edge_index)

        c_skip, c_out = [
            self.append_dims(x, model_output.ndim)
            for x in self.get_scalings_for_boundary_condition(t)
        ]

        if not model.sparse:
            x0 = x0.permute(0, 3, 1, 2)
        else:
            x0 = x0.reshape((-1, 2))
        denoise = c_out * model_output + c_skip * x0

        return model_output, denoise

    def consistency_test_step(self, model, batch, batch_idx, split="test"):
        t_start = time.time()
        edge_index = None
        np_edge_index = None
        device = batch[-1].device
        original_edge_index = None

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if not model.sparse:
            real_batch_idx, points, adj_matrix, gt_tour = batch
            np_points = points.cpu().numpy()[0]
            np_gt_tour = gt_tour.cpu().numpy()[0]
        else:
            real_batch_idx, graph_data, point_indicator, _, gt_tour = batch
            route_edge_flags = graph_data.edge_attr
            points = graph_data.x
            edge_index = graph_data.edge_index
            original_edge_index = edge_index.clone()
            num_edges = edge_index.shape[1]
            batch_size = point_indicator.shape[0]
            adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
            points = points.reshape((-1, 2))
            edge_index = edge_index.reshape((2, -1))
            np_points = points.cpu().numpy()
            np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
            np_edge_index = edge_index.cpu().numpy()

        if model.args.parallel_sampling > 1:
            if not model.sparse:
                points = points.repeat(model.args.parallel_sampling, 1, 1)
            else:
                points = points.repeat(model.args.parallel_sampling, 1)
                edge_index = model.duplicate_edge_index(
                    model.args.parallel_sampling, edge_index, np_points.shape[0], device
                )

        np_points_float64 = np_points.astype("float64")
        tsp_solver = TSPEvaluator(np_points)
        gt_cost = tsp_solver.evaluate(np_gt_tour)
        alpha = model.args.m2
        candidate_alpha = float(model.args.m1)
        a = min(max(float(model.args.rewrite_ratio), 0.0), 1.0)
        n = model.args.rewrite_steps
        rewrite_iterations = getattr(model.args, "rewrite_iterations", 1)
        if n > 1:
            re_schedule = [a + i * (1.0 - a) / (n - 1) for i in range(n)]
        else:
            re_schedule = [a]
        save_trace = bool(getattr(model.args, "save_tsp_rewrite_trace", False))
        trace_milestones = (5, 10, 15, 20)
        trace_data = None
        if save_trace:
            trace_data = {
                "initial_step_tours": [],
                "initial_step_costs": [],
                "initial_step_edge_masks": [],
                "iteration_milestones": [],
                "iteration_tours": [],
                "iteration_costs": [],
            }

        def solve_from_xt(xt):
            x0_prob = self.denoise_step(model, points, xt, device, edge_index)
            adj_mat = x0_prob.float().cpu().detach().numpy() + 1e-6
            tours, _ = merge_sub_tours(
                adj_mat, np_points, np_edge_index,
                sparse_graph=model.sparse,
                parallel_sampling=model.args.parallel_sampling,
                alpha=float(alpha),
                seed=getattr(model.args, "stitch_seed", None),
            )
            solved_tours, _ = batched_two_opt_torch(
                np_points_float64,
                np.array(tours).astype("int64"),
                max_iterations=model.args.two_opt_iterations,
                device=device,
            )
            solved_costs = [
                tsp_solver.evaluate(solved_tours[i])
                for i in range(solved_tours.shape[0])
            ]
            best_id = np.argmin(solved_costs)
            return x0_prob, solved_tours, solved_costs[best_id], best_id

        def tour_to_rewrite_source(tour, candidate_mask):
            source = model.tour2adj(
                tour,
                np_points,
                model.sparse,
                model.args.sparse_factor,
                original_edge_index,
            )
            source = source.unsqueeze(0).to(device)
            if model.args.parallel_sampling > 1:
                if not model.sparse:
                    source = source.repeat(model.args.parallel_sampling, 1, 1)
                else:
                    source = source.repeat(model.args.parallel_sampling, 1)

            if model.sparse:
                source = source.reshape(-1)

            if candidate_mask is not None:
                source = source * candidate_mask

            return source.clamp(0, 1)

        def selected_edge_mask(mask, best_id):
            if model.sparse:
                if model.args.parallel_sampling > 1:
                    return mask.reshape(model.args.parallel_sampling, -1)[best_id].cpu().numpy()
                return mask.reshape(-1).cpu().numpy()

            if mask.ndim == 3:
                return mask[best_id].cpu().numpy()
            return mask.cpu().numpy()

        def build_solution(seed_tour=None, seed_candidate_mask=None, step_trace=None):
            #TODO: 能否在每一轮构建解时，都对节点特征进行如下的类似的翻转来增强随机性 
            # self.feature_list = [
            #         encode_fn(coords),
            #         encode_fn(-coords),
            #         encode_fn(reflect_x(coords)),
            #         encode_fn(reflect_y(coords)),
            #     ]
            #     coords = np.flip(coords, axis=-1)
            #     self.feature_list += [
            #         encode_fn(coords),
            #         encode_fn(-coords),
            #         encode_fn(reflect_x(coords)),
            #         encode_fn(reflect_y(coords)),
            #     ]
            if seed_tour is None:
                xt = torch.zeros_like(adj_matrix.float())
                if model.args.parallel_sampling > 1:
                    if not model.sparse:
                        xt = xt.repeat(model.args.parallel_sampling, 1, 1)
                    else:
                        xt = xt.repeat(model.args.parallel_sampling, 1)

                if model.sparse:
                    xt = xt.reshape(-1)

                x0_prob, solved_tours, _, best_id = solve_from_xt(xt)
                step_tour = solved_tours[best_id]
                candidate_mask = (
                    (x0_prob >= candidate_alpha).float()
                    if candidate_alpha > 0
                    else None
                )
                best_tour = step_tour
                best_solved_cost = float("inf")
                best_candidate_mask = candidate_mask
                g_xt = None
            else:
                step_tour = seed_tour
                candidate_mask = seed_candidate_mask
                best_tour = seed_tour
                best_solved_cost = float("inf")
                best_candidate_mask = seed_candidate_mask
                g_xt = None

            for re in range(n):
                rewrite_source = tour_to_rewrite_source(step_tour, candidate_mask)
                if re == 0:
                    g_xt = torch.bernoulli((rewrite_source * re_schedule[0]).clamp(0, 1))
                else:
                    prev_ratio = re_schedule[re - 1]
                    cur_ratio = re_schedule[re]
                    add_ratio = (cur_ratio - prev_ratio) / max(1.0 - prev_ratio, 1e-12)
                    g_xt = torch.maximum(
                        g_xt,
                        torch.bernoulli(
                            ((1 - g_xt) * rewrite_source * add_ratio).clamp(0, 1)
                        ),
                    )

                x0_prob, solved_tours, step_cost, best_id = solve_from_xt(g_xt)
                step_tour = solved_tours[best_id]
                candidate_mask = (
                    (x0_prob >= candidate_alpha).float()
                    if candidate_alpha > 0
                    else None
                )

                if step_cost < best_solved_cost:
                    best_solved_cost = step_cost
                    best_tour = step_tour
                    best_candidate_mask = candidate_mask

                if step_trace is not None:
                    step_trace["tours"].append(np.array(step_tour, copy=True))
                    step_trace["costs"].append(float(step_cost))
                    step_trace["edge_masks"].append(selected_edge_mask(g_xt, best_id))

            return best_tour, best_solved_cost, best_candidate_mask

        stacked_tours = []
        stacked_costs = []
        stacked_candidate_masks = []

        # for _ in range(model.args.sequential_sampling):
        initial_step_trace = {"tours": [], "costs": [], "edge_masks": []} if save_trace else None
        solved_tour, solved_cost, solved_candidate_mask = build_solution(
            step_trace=initial_step_trace
        )
        if save_trace:
            trace_data["initial_step_tours"] = initial_step_trace["tours"]
            trace_data["initial_step_costs"] = initial_step_trace["costs"]
            trace_data["initial_step_edge_masks"] = initial_step_trace["edge_masks"]
        stacked_tours.append(solved_tour[None, :])
        stacked_costs.append(solved_cost)
        stacked_candidate_masks.append(solved_candidate_mask)

        solved_tours = np.concatenate(stacked_tours, axis=0)
        solved_costs = np.array(stacked_costs)
        best_solved_cost, best_id = np.min(solved_costs), np.argmin(solved_costs)
        gap = (best_solved_cost - gt_cost) / gt_cost * 100

        g_best_tour = solved_tours[best_id]
        candidate_mask = stacked_candidate_masks[best_id]

        rewrite_solved_cost = best_solved_cost
        rewrite_gap = gap

        for iteration_idx in range(1, rewrite_iterations + 1):
            step_tour, rewrite_solved_cost_tmp, step_candidate_mask = build_solution(
                g_best_tour, candidate_mask
            )
            if rewrite_solved_cost_tmp < rewrite_solved_cost:
                rewrite_solved_cost = rewrite_solved_cost_tmp
                g_best_tour = step_tour
                candidate_mask = step_candidate_mask

            rewrite_gap = (rewrite_solved_cost - gt_cost) / gt_cost * 100
            if save_trace and iteration_idx in trace_milestones:
                trace_data["iteration_milestones"].append(iteration_idx)
                trace_data["iteration_tours"].append(np.array(g_best_tour, copy=True))
                trace_data["iteration_costs"].append(float(rewrite_solved_cost))

        if save_trace:
            trace_dir = getattr(model.args, "tsp_rewrite_trace_dir", None)
            if trace_dir is None:
                exp_save_dir = os.path.join(
                    model.logger.save_dir, model.logger.name, model.logger.version
                )
                trace_dir = os.path.join(exp_save_dir, "tsp_rewrite_trace")
            os.makedirs(trace_dir, exist_ok=True)
            instance_id = int(real_batch_idx.cpu().numpy().reshape(-1)[0])
            np.savez_compressed(
                os.path.join(trace_dir, f"{split}-instance-{instance_id}.npz"),
                points=np_points,
                edge_index=(
                    np_edge_index
                    if np_edge_index is not None
                    else np.empty((2, 0), dtype=np.int64)
                ),
                sparse_graph=bool(model.sparse),
                gt_tour=np_gt_tour,
                gt_cost=float(gt_cost),
                initial_step_tours=np.asarray(trace_data["initial_step_tours"], dtype=np.int64),
                initial_step_costs=np.asarray(trace_data["initial_step_costs"], dtype=np.float64),
                initial_step_edge_masks=np.asarray(
                    trace_data["initial_step_edge_masks"], dtype=np.float32
                ),
                initial_best_tour=np.asarray(solved_tours[best_id], dtype=np.int64),
                initial_best_cost=float(best_solved_cost),
                rewrite_iteration_milestones=np.asarray(
                    trace_data["iteration_milestones"], dtype=np.int64
                ),
                rewrite_iteration_tours=np.asarray(trace_data["iteration_tours"], dtype=np.int64),
                rewrite_iteration_costs=np.asarray(trace_data["iteration_costs"], dtype=np.float64),
                final_rewrite_tour=np.asarray(g_best_tour, dtype=np.int64),
                final_rewrite_cost=float(rewrite_solved_cost),
                rewrite_steps=int(n),
                rewrite_iterations=int(rewrite_iterations),
                rewrite_ratio=float(model.args.rewrite_ratio),
                m1=float(model.args.m1),
                m2=float(model.args.m2),
            )

        if torch.cuda.is_available() and device.type == "cuda":
            gpu_mem = torch.cuda.memory_allocated(device) / 1024 ** 2
            gpu_max_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        else:
            gpu_mem = 0.0
            gpu_max_mem = 0.0

        metrics = {
            f"{split}/rewrite_ratio": float(model.args.rewrite_ratio),
            f"{split}/rewrite_iterations": float(rewrite_iterations),
            f"{split}/gap": gap,
            f"{split}/rewrite_gap": rewrite_gap,
            f"{split}/gt_cost": gt_cost,
            f"{split}/rewrite_solved_cost": rewrite_solved_cost,
            f"{split}/time": time.time() - t_start,
            f"{split}/gpu_memory_MB": gpu_mem,
            f"{split}/gpu_max_memory_MB": gpu_max_mem,
        }

        for k, v in metrics.items():
            model.log(k, v, on_epoch=True, sync_dist=True)
        model.log(
            f"{split}/solved_cost",
            best_solved_cost,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )

        return metrics

    def denoise_step(self, model, points, xt, device, edge_index=None):
        with torch.no_grad():
            xt = xt.float()
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (
                1.0 + 0.05 * torch.rand_like(xt.float(), device=device)
            )

            x0_pred = model.forward(
                points.float().to(device),
                xt_scale.to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )

            if not model.sparse:
                x0_pred_prob = (
                    x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
                )
            else:
                x0_pred_prob = x0_pred.reshape((-1, 2)).softmax(dim=-1)

            return x0_pred_prob[..., 1]
