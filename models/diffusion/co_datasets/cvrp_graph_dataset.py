import torch
import numpy as np
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData



class CVRPGraphDataset(Dataset):
    def __init__(self, data_file, sparse_factor=-1, graph_type="undirected"):
        self.data_file = data_file
        self.sparse_factor = sparse_factor
        self.graph_type = graph_type

        with open(data_file, "r") as file:
            self.file_lines = file.read().splitlines()

        print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

    def __len__(self):
        return len(self.file_lines)

    def get_example(self, idx):
        line = self.file_lines[idx].strip()

        split_line_0 = line.split("depots ")[1]
        split_line_1 = split_line_0.split(" points ")
        depot = np.array([float(x) for x in split_line_1[0].split(" ")])

        split_line_2 = split_line_1[1].split(" demands ")
        points = np.array([[float(split_line_2[0].split(" ")[i]), float(split_line_2[0].split(" ")[i + 1])]
                           for i in range(0, len(split_line_2[0].split(" ")), 2)])

        split_line_3 = split_line_2[1].split(" capacity ")
        demands = np.array([float(x) for x in split_line_3[0].split(" ")])

        split_line_4 = split_line_3[1].split(" output ")
        capacity = float(split_line_4[0])
        tour = np.array([int(x) for x in split_line_4[1].split(" ")])

        # 截断 -1
        if tour is not None and -1 in tour:
            tour = tour[:np.where(tour == -1)[0][0]]

        return depot, points, demands, capacity, tour

    def __getitem__(self, idx):
        depot, points, demands, capacity, tour = self.get_example(idx)

        nodes_num = len(points) + 1

        # merge depot and points into node features
        nodes = np.vstack([depot[None, :], points])  # shape: [N+1, 2]
        demands = demands / capacity
        demands = np.concatenate([[0.0], demands]).reshape(nodes_num, 1)    # depot has 0 demand

        depot_flag = np.zeros(shape=(nodes_num,))
        depot_flag[0] = 1
        depot_flag = depot_flag.reshape(nodes_num, 1)
        x = np.concatenate([nodes, demands, depot_flag], 1)

        if self.sparse_factor <= 0:
            adj_matrix = np.zeros((nodes_num, nodes_num))
            dis_matrix = cdist(nodes, nodes)
            for i in range(len(tour) - 1):
                u, v = tour[i], tour[i + 1]
                adj_matrix[u, v] = 1
                if self.graph_type == "undirected":
                    adj_matrix[v, u] = 1

            tourdata = GraphData(
                tour = torch.from_numpy(tour).long(),
                tour_len=torch.tensor([len(tour)], dtype=torch.long)
            )

            return (
                torch.LongTensor([idx]),
                torch.from_numpy(x).float(),
                # torch.from_numpy(demands).float(),
                # torch.tensor(capacity).float(),
                torch.from_numpy(dis_matrix).float(),
                torch.from_numpy(adj_matrix).float(),
                tourdata
            )
        else:
            # use k-NN graph
            kdt = KDTree(nodes, leaf_size=30, metric="euclidean")
            _, idx_knn = kdt.query(nodes, k=self.sparse_factor)

            edge_index_0 = torch.arange(nodes.shape[0]).view(-1, 1).repeat(1, self.sparse_factor).reshape(-1)
            edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))
            edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

            graph_data = GraphData(
                x=torch.from_numpy(nodes).float(),
                edge_index=edge_index,
                edge_attr=None  # optionally encode edge weight or tour info
            )

            point_indicator = torch.tensor([nodes.shape[0]]).long()
            edge_indicator = torch.tensor([edge_index.shape[1]]).long()

            return (
                torch.LongTensor([idx]),
                graph_data,
                torch.from_numpy(demands).float(),
                torch.tensor(capacity).float(),
                point_indicator,
                edge_indicator,
                torch.from_numpy(tour).long()
            )
