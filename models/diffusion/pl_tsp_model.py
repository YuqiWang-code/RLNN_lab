import os

import numpy as np
import torch

from diffusion.co_datasets.tsp_graph_dataset import TSPGraphDataset
from diffusion.consistency import TSPConsistency
from diffusion.pl_meta_model import COMetaModel


class TSPModel(COMetaModel):
    def __init__(self, param_args=None, load_dataset=True):
        super(TSPModel, self).__init__(param_args=param_args, node_feature_only=False)

        if load_dataset:
            if self.args.training_split:
                self.train_dataset = TSPGraphDataset(
                    data_file=os.path.join(self.args.training_split),
                    sparse_factor=self.args.sparse_factor,
                    graph_type=self.args.graph_type,
                )
            if self.args.test_split:
                self.test_dataset = TSPGraphDataset(
                    data_file=os.path.join(self.args.test_split),
                    sparse_factor=self.args.sparse_factor,
                    graph_type=self.args.graph_type,
                )
            if self.args.validation_split:
                self.validation_dataset = TSPGraphDataset(
                    data_file=os.path.join(self.args.validation_split),
                    sparse_factor=self.args.sparse_factor,
                    graph_type=self.args.graph_type,
                )

        self.consistency_trainer = TSPConsistency(
            self.args,
            sigma_max=self.diffusion.T,
            boundary_func=self.args.boundary_func,
        )

    def forward(self, x, adj, t=None, edge_index=None):
        if edge_index is None and t is not None and getattr(t, "dim", lambda: 0)() == 2:
            edge_index = t
        logits = self.model(x, adj, edge_index)
        if logits.dim() == 3:
            logits = torch.stack((-logits, logits), dim=1)
        return logits

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
            adj_matrix = torch.zeros((points.shape[0], points.shape[0]))
            for i in range(tour.shape[0] - 1):
                adj_matrix[tour[i], tour[i + 1]] = 1
                adj_matrix[tour[i + 1], tour[i]] = 1
        else:
            adj_matrix = np.zeros(points.shape[0], dtype=np.int64)
            adj_matrix[tour[:-1]] = tour[1:]
            adj_matrix = torch.from_numpy(adj_matrix)
            adj_matrix = (
                adj_matrix.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
            )
            adj_matrix = torch.eq(edge_index[1].cpu(), adj_matrix).to(torch.int)
        return adj_matrix
