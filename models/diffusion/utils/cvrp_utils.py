import copy
import numpy as np
import scipy.sparse
import scipy.spatial
from torch import Tensor
from typing import Union, List
from diffusion.utils.local_search.classic import cvrp_classic_local_search


def merge_tours(
    adj_mat, np_points, edge_index_np, sparse_graph=False, parallel_sampling=1, use_ls=False
):
    if not sparse_graph:
        return cvrp_dense_decode(adj_mat, np_points, use_ls)


def cvrp_dense_decode(
    heatmap: np.ndarray, nodes_feature: np.ndarray, use_ls: bool = False
) -> Union[List[np.ndarray], np.floating]:
    # heatmap
    # heatmap: np.ndarray = to_numpy(vars)
    # nodes_feature: np.ndarray = to_numpy(nodes_feature)
    depots = nodes_feature[:, 0, :2]
    points = nodes_feature[:, 1:, :2]
    demands = nodes_feature[:, :, 2]
    heatmap = (heatmap + heatmap.transpose(0, 2, 1)) / 2

    # decoding
    tours = list()
    for idx in range(len(heatmap)):
        tour = cvrp_greedy_decoder(
            heatmap=copy.deepcopy(heatmap[idx]), norm_demand=demands[idx]
        )
        if use_ls:
            tour = cvrp_classic_local_search(
                init_tour=tour, depot=depots[idx],
                points=points[idx], demands=demands[idx]
            )
        tours.append(tour)

    return tours

def cvrp_sparse_decode(
        vars: Tensor, nodes_feature: Tensor, edge_index: Tensor, nodes_num_list: list,
        edges_num_list: list, return_cost: bool, use_ls: bool = True
) -> Union[List[np.ndarray], np.floating]:
    # preparation
    device = vars.device
    heatmap = to_numpy(vars)
    edge_index = to_numpy(edge_index)
    nodes_feature = to_numpy(nodes_feature)
    points = nodes_feature[1:, :2]
    depots = nodes_feature[0, :2]
    demands = nodes_feature[:, 2]

    # from heatmap to decoding the tours
    tours = list()
    x_begin_idx = 0
    e_begin_idx = 0
    for nodes_num, edges_num in zip(nodes_num_list, edges_num_list):
        # preparation
        x_end_idx = x_begin_idx + nodes_num
        e_end_idx = e_begin_idx + edges_num
        _heatmap = heatmap[e_begin_idx:e_end_idx]
        _points = points[x_begin_idx:x_end_idx]
        _edge_index = edge_index[:, e_begin_idx:e_end_idx] - x_begin_idx
        _depots = depots[x_begin_idx:x_end_idx]
        _demands = demands[x_begin_idx:x_end_idx]



        # heatmap sparse to dense
        _heatmap_dense = np_sparse_to_dense(
            nodes_num=nodes_num, edge_index=_edge_index, edge_attr=_heatmap
        )
        _heatmap = (_heatmap_dense + _heatmap_dense.T) / 2

        # greedy decoding
        greedy_tour = cvrp_greedy_decoder(heatmap=copy.deepcopy(_heatmap), norm_demand=_demands)

        # 2OPT
        if use_ls:
            greedy_tour = cvrp_classic_local_search(
                init_tour=greedy_tour, depot=_depots,
                points=_points, demands=_demands
            )

        # update index
        x_begin_idx = x_end_idx
        e_begin_idx = e_end_idx
        tours.append(greedy_tour)

    # format
    tours = np.array(tours)

    if return_cost:
        tmp_solver = CVRPSolver()
        tmp_solver.from_data(
            depots=depots,
            points=points,
            demands=demands[1:],
            capacities=np.ones(shape=(len(tours),)),
            tours=tours
        )
        return tmp_solver.evaluate()
    else:
        return tours


def cvrp_greedy_decoder(
        heatmap: np.ndarray, norm_demand: np.ndarray, cp_alpha: float = 0.5
) -> np.ndarray:
    # prepare for decoding
    np.fill_diagonal(heatmap, 0)
    current_capacity = 1.0
    nodes_visited = np.zeros_like(norm_demand).astype(np.bool_)
    tour = [0]
    current_node = 0
    nodes_visited[0] = True

    # greedy search
    while not nodes_visited.all():
        # find next node
        if current_capacity >= cp_alpha:
            next_node = np.argmax(heatmap[current_node][1:]) + 1
        else:
            next_node = np.argmax(heatmap[current_node])
            if norm_demand[next_node] > current_capacity:
                next_node = 0

        # update
        nodes_visited[next_node] = True
        tour.append(next_node)
        if next_node == 0:
            current_capacity = 1.0
        else:
            current_capacity -= norm_demand[next_node]
            heatmap[:, next_node] = -1
        current_node = next_node

    # return
    tour.append(0)
    return np.array(tour)

class TSPEvaluator(object):
    def __init__(self, dis_matrix):
        self.dist_mat = dis_matrix

    def evaluate(self, route):
        total_cost = 0
        for i in range(len(route) - 1):
            total_cost += self.dist_mat[route[i], route[i + 1]]
        return total_cost