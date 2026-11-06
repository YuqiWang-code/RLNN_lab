import os

import numpy as np
import torch

from diffusion.co_datasets.cvrp_graph_dataset import CVRPGraphDataset
from diffusion.consistency import CVRPConsistency
from diffusion.models.gnn_cvrp_encoder import GNNEncoder
from diffusion.pl_meta_model import COMetaModel


class CVRPModel(COMetaModel):
    def __init__(self, param_args=None, load_dataset=True):
        super(CVRPModel, self).__init__(param_args=param_args, node_feature_only=False)

        self.model = GNNEncoder(
            n_layers=self.args.n_layers,
            hidden_dim=self.args.hidden_dim,
            out_channels=2,
            aggregation=self.args.aggregation,
            sparse=self.sparse,
            use_activation_checkpoint=self.args.use_activation_checkpoint,
            node_feature_only=False,
        )

        if load_dataset:
            if self.args.training_split:
                self.train_dataset = CVRPGraphDataset(
                    data_file=os.path.join(self.args.training_split),
                    sparse_factor=self.args.sparse_factor,
                    graph_type=self.args.graph_type,
                )
            if self.args.test_split:
                self.test_dataset = CVRPGraphDataset(
                    data_file=os.path.join(self.args.test_split),
                    sparse_factor=self.args.sparse_factor,
                    graph_type=self.args.graph_type,
                )
            if self.args.validation_split:
                self.validation_dataset = CVRPGraphDataset(
                    data_file=os.path.join(self.args.validation_split),
                    sparse_factor=self.args.sparse_factor,
                    graph_type=self.args.graph_type,
                )

        self.consistency_trainer = CVRPConsistency(
            self.args,
            sigma_max=self.diffusion.T,
            boundary_func=self.args.boundary_func,
        )

    def forward(self, x, adj, dis_matrix, edge_index):
        return self.model(x, adj, dis_matrix, edge_index)

    def training_step(self, batch, batch_idx):
        loss = self.consistency_trainer.consistency_losses(self, batch)
        self.log("train/loss", loss)
        return loss

    def test_step(self, batch, batch_idx, split="test"):
        return self.consistency_trainer.consistency_test_step(
            self, batch, batch_idx, split
        )

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx, split="val")

    def tour2adj(self, tour, points, sparse, sparse_factor, edge_index):
        if not sparse:
            adj_matrix = torch.zeros((points, points))
            for i in range(tour.shape[0] - 1):
                adj_matrix[tour[i], tour[i + 1]] = 1
                adj_matrix[tour[i + 1], tour[i]] = 1
        else:
            adj_matrix = np.zeros(points, dtype=np.int64)
            adj_matrix[tour[:-1]] = tour[1:]
            adj_matrix = torch.from_numpy(adj_matrix)
            adj_matrix = (
                adj_matrix.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            )
            adj_matrix = torch.eq(edge_index[1].cpu(), adj_matrix).to(torch.int)
        return adj_matrix

    def points2adj(self, points):
        assert points.dim() == 3
        return (
            torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1) ** 0.5
        )
