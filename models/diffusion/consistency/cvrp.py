import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from diffusion.consistency.meta import MetaConsistency
from diffusion.utils.cvrp_utils import TSPEvaluator, merge_tours

import time


class CVRPConsistency(MetaConsistency):
    def __init__(
        self,
        args,
        sigma_max=1000,
        sigma_min=0,
        weight_schedule="uniform",
        boundary_func="truncate",
    ):
        super(CVRPConsistency, self).__init__(
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            weight_schedule=weight_schedule,
            boundary_func=boundary_func,
        )
        self.args = args

    def consistency_losses(self, model, batch):
        edge_index = None
        if model.sparse:
            idx, points, dis_matrix, adj_matrix, tourdata = batch
        else:
            idx, points, dis_matrix, adj_matrix, tourdata = batch

        t = torch.randint(
                1, model.diffusion.T + 1, [idx.shape[0]], device=points.device
            )
        x0 = F.one_hot(
            adj_matrix.long(), num_classes=2
        ).float()  # B, N, N, 2 if not sparse else B, N*N, 2
        if model.sparse:
            x0 = x0.unsqueeze(1)  # B, 1, N*K, 2

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

        _, denoise = self.denoise(model, points, x_t, t, dis_matrix, edge_index, x0)
        adj_matrix = adj_matrix.long()
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(denoise, adj_matrix)
        if model.args.consistency:
            _, denoise2 = self.denoise(
                model, points, x_t2, t2, dis_matrix, edge_index, x0
            )
            loss = loss + loss_func(denoise2, adj_matrix)

        return loss

    def denoise(self, model, points, x_t, t, dis_matrix, edge_index, x0):
        x_t = x_t * 2 - 1
        x_t = x_t * (1.0 + 0.05 * torch.rand_like(x_t))
        model_output = model(
            points, x_t, dis_matrix, edge_index
        )  # B, 2, N, N if not sparse else B*N*K, 2; k is the sparse factor

        c_skip, c_out = [
            self.append_dims(x, model_output.ndim)
            for x in self.get_scalings_for_boundary_condition(t)
        ]  # B, 1, 1, 1 or B*N*K, 1

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
        device = batch[0].device
        original_edge_index = None

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        if not model.sparse:
            idx, points, dis_matrix, adj_matrix, tourdata = batch
            tour_lens = tourdata.tour_len.view(-1).tolist()  # [B]
            gt_tour = torch.split(tourdata.tour, tour_lens)
            np_points = points.cpu().numpy()
            np_gt_tour = gt_tour[0].cpu().numpy()
            np_dis_matrix = dis_matrix.cpu().numpy()[0]
        else:
            idx, graph_data, demands, adj_matrix, capacity, point_indicator, edge_indicator, gt_tour = batch
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

        tsp_solver = TSPEvaluator(np_dis_matrix)
        gt_cost = tsp_solver.evaluate(np_gt_tour)
        candidate_alpha = float(model.args.m1)
        a = min(max(float(model.args.rewrite_ratio), 0.0), 1.0)
        n = model.args.rewrite_steps
        rewrite_iterations = getattr(model.args, "rewrite_iterations", 1)
        if n > 1:
            re_schedule = [a + i * (1.0 - a) / (n - 1) for i in range(n)]
        else:
            re_schedule = [a]

        def solve_from_xt(xt):
            x0_prob = self.denoise_step(model, points, xt, dis_matrix, device, edge_index)
            adj_mat = x0_prob.float().cpu().detach().numpy() + 1e-6
            solved_tours = merge_tours(
                adj_mat,
                np_points,
                np_edge_index,
                sparse_graph=model.sparse,
                parallel_sampling=model.args.parallel_sampling,
                use_ls=model.args.two_opt_iterations,
            )
            solved_tours = np.array(solved_tours)
            solved_costs = [
                tsp_solver.evaluate(solved_tours[i])
                for i in range(solved_tours.shape[0])
            ]
            best_id = np.argmin(solved_costs)
            return x0_prob, solved_tours, solved_costs[best_id], best_id

        def tour_to_rewrite_source(tour, candidate_mask):
            source = model.tour2adj(
                tour,
                np_points.shape[1],
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

        def build_solution(seed_tour=None, seed_candidate_mask=None):
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

            return best_tour, best_solved_cost, best_candidate_mask

        stacked_tours = []
        stacked_costs = []
        stacked_candidate_masks = []

        solved_tour, solved_cost, solved_candidate_mask = build_solution()
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

        for _ in range(rewrite_iterations):
            step_tour, rewrite_solved_cost_tmp, step_candidate_mask = build_solution(
                g_best_tour, candidate_mask
            )
            if rewrite_solved_cost_tmp < rewrite_solved_cost:
                rewrite_solved_cost = rewrite_solved_cost_tmp
                g_best_tour = step_tour
                candidate_mask = step_candidate_mask

            rewrite_gap = (rewrite_solved_cost - gt_cost) / gt_cost * 100

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

    def denoise_step(self, model, points, xt, dis_matrix, device, edge_index=None):
        with torch.no_grad():
            xt = xt.float()  # b, n, n
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (
                1.0 + 0.05 * torch.rand_like(xt.float(), device=device)
            )

            # [b, 2, n, n]
            x0_pred = model.forward(
                points.float().to(device),
                xt_scale.to(device),
                dis_matrix,
                edge_index.long().to(device) if edge_index is not None else None,
            )

            if not model.sparse:
                x0_pred_prob = (
                    x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
                )
            else:
                x0_pred_prob = x0_pred.reshape((-1, 2)).softmax(dim=-1)

            return x0_pred_prob[..., 1]

    def guided_denoise_step(
        self, model, points, xt_prob, x0, t, device, edge_index=None
    ):
        # xt_prob [B, N, N, 2]
        torch.set_grad_enabled(True)

        # straight-through gumbel-Softmax
        xt_prob.requires_grad = True
        xt = xt_prob[..., 1].float()
        # xt = F.gumbel_softmax(xt_prob, tau=1, hard=True, dim=-1)[...,1]
        xt = xt.float()

        # b, n, n
        # xt.requires_grad = True

        with torch.inference_mode(False):
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (
                1.0 + 0.05 * torch.rand_like(xt.float(), device=device)
            )
            # [b, 2, n, n]
            x0_pred = model.forward(
                points.float().to(device),
                xt_scale.to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )

            # Compute loss
            loss_func = nn.CrossEntropyLoss()
            bce_loss = loss_func(x0_pred, x0.long())

            if not model.sparse:
                p_theta = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
            else:
                p_theta = x0_pred.reshape((-1, 2)).softmax(dim=-1)

            if not model.sparse:
                dis_matrix = model.points2adj(points)
                cost_est = (dis_matrix * p_theta[..., 1]).sum()
                # cost_est.requires_grad_(True)
                # cost_est.backward()
            else:
                dis_matrix = torch.sqrt(
                    torch.sum(
                        (points[edge_index.T[:, 0]] - points[edge_index.T[:, 1]]) ** 2,
                        dim=1,
                    )
                )

                cost_est = (dis_matrix * p_theta[..., 1]).sum()
                # cost_est.requires_grad_(True)
                # cost_est.backward()

            loss = self.args.c1 * bce_loss + self.args.c2 * cost_est
            loss.backward()
            # if model.args.norm is True:
            #     xt.grad = torch.nn.functional.normalize(xt.grad, p=2, dim=-1)
        torch.set_grad_enabled(False)
        assert xt_prob.grad is not None

        # compute p_phi
        with torch.no_grad():
            # scale = 50
            p_phi = torch.exp(-xt_prob.grad)
            if model.sparse:
                p_phi = p_phi.reshape(p_theta.shape)
            xt_prob_u = (xt_prob * p_phi) / torch.sum(
                (xt_prob * p_phi), dim=-1, keepdim=True
            )

            xt_u = torch.bernoulli(xt_prob_u[..., 1].clamp(0, 1))
            xt_u = xt_u.float()  # b, n, n
            xt_scale_u = xt_u * 2 - 1
            xt_scale_u = xt_scale_u * (
                1.0 + 0.05 * torch.rand_like(xt_u.float(), device=device)
            )       # b*n

            # [b, 2, n, n]
            x0_pred_u = model.forward(
                points.float().to(device),
                xt_scale_u.to(device),
                t.float().to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )

            if not model.sparse:
                x0_pred_prob_u = (
                    x0_pred_u.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
                )
            else:
                x0_pred_prob_u = x0_pred_u.reshape((-1, 2)).softmax(dim=-1)
            output = torch.concat((p_theta[..., 1], x0_pred_prob_u[..., 1]), dim=0)
            # output = torch.concat((x0_pred_prob_u[..., 1], x0_pred_prob_u[..., 1]), dim=0)
            # output = torch.concat((p_theta[..., 1], p_theta[..., 1]), dim=0)
            return output
