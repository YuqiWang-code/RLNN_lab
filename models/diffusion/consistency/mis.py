import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from torch_sparse import SparseTensor
from diffusion.consistency.meta import MetaConsistency
from diffusion.utils.mis_utils import mis_decode_np

import time


class MISConsistency(MetaConsistency):
    def __init__(
            self,
            args,
            sigma_max=1000,
            sigma_min=0,
            weight_schedule="uniform",
            boundary_func='truncate'
    ):
        super(MISConsistency, self).__init__(
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            weight_schedule=weight_schedule,
            boundary_func=boundary_func)

        self.args = args

    def consistency_losses(self, model, batch):
        _, graph_data, point_indicator = batch  # point_indicator: B, 1, number of nodes in each batch
        node_labels = graph_data.x  # N*B
        edge_index = graph_data.edge_index  # 2, E*B

        x0 = F.one_hot(node_labels.long(), num_classes=2).float()

        device = node_labels.device
        t = torch.randint(1, model.diffusion.T + 1, [point_indicator.shape[0]]).to(device)
        t = t.repeat_interleave(point_indicator.reshape(-1), dim=0)

        if model.args.consistency:
            t2 = (model.args.alpha * t).int().to(device)
            x_t, x_t2 = model.diffusion.consistency_sample(
                x0.unsqueeze(1).unsqueeze(1),
                t.cpu().numpy(),
                t2.cpu().numpy(),
            )
        else:
            x_t, _ = model.diffusion.sample(x0.unsqueeze(1).unsqueeze(1), t)
            t2 = None
            x_t2 = None

        t = t.reshape(-1).float()
        x_t = x_t.reshape(-1).to(device)  # N
        edge_index = edge_index.to(device).reshape(2, -1)  # 2, E

        model_output, denoise = self.denoise(model, x_t, t, edge_index, x0)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(denoise, node_labels)
        if model.args.consistency:
            t2 = t2.reshape(-1).float()
            x_t2 = x_t2.reshape(-1).to(device)
            _, denoise2 = self.denoise(model, x_t2, t2, edge_index, x0)
            loss = loss + loss_func(denoise2, node_labels)

        return loss

    def denoise(self, model, x_t, t, edge_index, x0):
        x_t = x_t * 2 - 1
        x_t = x_t * (1.0 + 0.05 * torch.rand_like(x_t))
        model_output = model(x_t, edge_index)

        c_skip, c_out = [
            self.append_dims(x, model_output.ndim)
            for x in self.get_scalings_for_boundary_condition(t)
        ]

        denoise = c_out * model_output + c_skip * x0

        return model_output, denoise

    def consistency_test_step(self, model, batch, batch_idx, split='test'):
        device = batch[-1].device
        t_start = time.time()

        real_batch_idx, graph_data, point_indicator = batch
        node_labels = graph_data.x
        edge_index = graph_data.edge_index

        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        edge_index = edge_index.to(node_labels.device).reshape(2, -1)
        edge_index_np = edge_index.cpu().numpy()
        adj_mat = scipy.sparse.coo_matrix(
            (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1])),
        )

        if model.args.parallel_sampling > 1:
            edge_index = model.duplicate_edge_index(model.args.parallel_sampling, edge_index, node_labels.shape[0], device)

        candidate_alpha = float(model.args.m1)
        a = min(max(float(model.args.rewrite_ratio), 0.0), 1.0)
        n = model.args.rewrite_steps
        rewrite_iterations = getattr(model.args, "rewrite_iterations", 1)
        if n > 1:
            re_schedule = [a + i * (1.0 - a) / (n - 1) for i in range(n)]
        else:
            re_schedule = [a]

        def solve_from_xt(xt):
            x0_prob = self.denoise_step(model, xt, device, edge_index)
            predict_labels = x0_prob.float().cpu().detach().numpy().reshape(-1) + 1e-6
            split_predict_labels = np.split(
                predict_labels,
                model.args.parallel_sampling,
            )
            solved_solutions = [
                mis_decode_np(predict_labels, adj_mat, model.args.ep)
                for predict_labels in split_predict_labels
            ]
            solved_costs = [solved_solution.sum() for solved_solution in solved_solutions]
            best_id = np.argmax(solved_costs)
            return x0_prob, solved_solutions, solved_costs[best_id], best_id

        def solution_to_rewrite_source(solution, candidate_mask):
            source = torch.from_numpy(solution).float().to(device)
            if model.args.parallel_sampling > 1:
                source = source.repeat(model.args.parallel_sampling)

            source = source.reshape(-1)
            if candidate_mask is not None:
                source = source * candidate_mask.reshape(-1)

            return source.clamp(0, 1)

        def build_solution(seed_solution=None, seed_candidate_mask=None):
            if seed_solution is None:
                xt = torch.zeros_like(node_labels.float())
                if model.args.parallel_sampling > 1:
                    xt = xt.repeat(model.args.parallel_sampling)

                xt = xt.reshape(-1)
                x0_prob, solved_solutions, _, best_id = solve_from_xt(xt)
                step_solution = solved_solutions[best_id]
                candidate_mask = (
                    (x0_prob >= candidate_alpha).float()
                    if candidate_alpha > 0
                    else None
                )
                best_solution = step_solution
                best_solved_cost = float("-inf")
                best_candidate_mask = candidate_mask
                g_xt = None
            else:
                step_solution = seed_solution
                candidate_mask = seed_candidate_mask
                best_solution = seed_solution
                best_solved_cost = float("-inf")
                best_candidate_mask = seed_candidate_mask
                g_xt = None

            for re in range(n):
                rewrite_source = solution_to_rewrite_source(step_solution, candidate_mask)
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

                x0_prob, solved_solutions, step_cost, best_id = solve_from_xt(g_xt)
                step_solution = solved_solutions[best_id]
                candidate_mask = (
                    (x0_prob >= candidate_alpha).float()
                    if candidate_alpha > 0
                    else None
                )

                if step_cost > best_solved_cost:
                    best_solved_cost = step_cost
                    best_solution = step_solution
                    best_candidate_mask = candidate_mask

            return best_solution, best_solved_cost, best_candidate_mask

        stacked_solutions = []
        stacked_costs = []
        stacked_candidate_masks = []

        for _ in range(model.args.sequential_sampling):
            solved_solution, solved_cost, solved_candidate_mask = build_solution()
            stacked_solutions.append(solved_solution[None, :])
            stacked_costs.append(solved_cost)
            stacked_candidate_masks.append(solved_candidate_mask)

        solved_solutions = np.concatenate(stacked_solutions, axis=0)
        solved_costs = np.array(stacked_costs)
        best_solved_cost, best_solved_id = np.max(solved_costs), np.argmax(solved_costs)

        gt_cost = node_labels.cpu().numpy().sum()
        gap = (gt_cost - best_solved_cost) / gt_cost * 100

        g_best_solution = solved_solutions[best_solved_id]
        candidate_mask = stacked_candidate_masks[best_solved_id]
        rewrite_solved_cost = best_solved_cost
        rewrite_gap = gap

        for _ in range(rewrite_iterations):
            step_solution, rewrite_solved_cost_tmp, step_candidate_mask = build_solution(
                g_best_solution, candidate_mask
            )
            if rewrite_solved_cost_tmp > rewrite_solved_cost:
                rewrite_solved_cost = rewrite_solved_cost_tmp
                g_best_solution = step_solution
                candidate_mask = step_candidate_mask

            rewrite_gap = (gt_cost - rewrite_solved_cost) / gt_cost * 100

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
        model.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)

        return metrics

    def denoise_step(self, model, xt, device, edge_index=None):
        with torch.no_grad():
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (
                1.0 + 0.05 * torch.rand_like(xt.float(), device=device)
            )   # b*n (700)

            # [b*n, 2]
            x0_pred = model.forward(
                xt_scale.to(device),
                edge_index.long().to(device) if edge_index is not None else None,
            )

            x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)     # 1, b*n, 1, 2
            # x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)

            return x0_pred_prob[..., 1]

    def guided_denoise_step(self, model, xt_prob, x0, t, device, edge_index):
        torch.set_grad_enabled(True)

        xt_prob.requires_grad = True
        xt = xt_prob[..., 1].reshape(-1)    # b*n
        # xt.requires_grad = True

        with torch.inference_mode(False):
            xt_scale = xt * 2 - 1
            xt_scale = xt_scale * (1.0 + 0.05 * torch.rand_like(xt.float(), device=device))

            x0_pred = model(
                xt_scale,   # b*n
                edge_index.long().to(device) if edge_index is not None else None,   # 2, E(73742)
            )   # b*n, 2

            loss_func = nn.CrossEntropyLoss()

            bce_loss = loss_func(x0_pred, x0.long())

            p_theta = x0_pred.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)

            num_nodes = xt.shape[0]
            adj_matrix = SparseTensor(
                row=edge_index[0],
                col=edge_index[1],
                value=torch.ones_like(edge_index[0].float()),
                sparse_sizes=(num_nodes, num_nodes),
            ).to_dense()
            adj_matrix.fill_diagonal_(0)
            pred_nodes = p_theta[..., 1].squeeze(0)

            f_mis = -pred_nodes.sum()
            g_mis = adj_matrix @ pred_nodes
            g_mis = (pred_nodes * g_mis).sum()
            cost_est = f_mis + 0.5 * g_mis

            # loss = 2 * bce_loss + 2 * cost_est    # for er
            # loss = 1 * bce_loss + 10 * cost_est    # for sat
            loss = self.args.c1 * bce_loss + self.args.c2 * cost_est
            loss.backward()

        torch.set_grad_enabled(False)
        assert xt_prob.grad is not None

        with torch.no_grad():
            # xt_grad = (
            #     torch.zeros(xt.shape, device=xt.device).unsqueeze(-1).repeat(1, 1, 2)   # b, n, 2
            # )
            # xt_grad[..., 1] = xt.grad
            # xt_grad[..., 0] = -xt.grad

            # scale = 0.5
            xt_prob.grad = xt_prob.grad.clamp(-1, 1)    # without this, exp(grad) explodes!
            p_phi = torch.exp(-xt_prob.grad)     # b, n, 2
            xt_prob_u = (xt_prob * p_phi) / torch.sum(
                (xt_prob * p_phi + 1e-6), dim=-1, keepdim=True
            )

            xt_u = torch.bernoulli(xt_prob_u[..., 1].clamp(0, 1))
            xt_u = xt_u.float().reshape(-1)  # b * n
            xt_scale_u = xt_u * 2 - 1
            xt_scale_u = xt_scale_u * (
                    1.0 + 0.05 * torch.rand_like(xt_u.float(), device=device)
            )

            x0_pred_u = model.forward(
                xt_scale_u.to(device),  # b*n
                t.float().to(device),   # 1
                edge_index.long().to(device) if edge_index is not None else None,   # 2, E
            )

            x0_pred_prob_u = x0_pred_u.reshape((1, xt.shape[0], -1, 2)).softmax(dim=-1)
            output = torch.cat((p_theta[..., 1], x0_pred_prob_u[..., 1]), dim=0)

            return output
