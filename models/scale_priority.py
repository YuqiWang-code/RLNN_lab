"""A single-pass MIS node priority from the fused RLNN encoder."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models import GCN


def standardize(x):
    centered = x - x.mean(dim=0, keepdim=True)
    return centered * centered.square().mean(dim=0, keepdim=True).clamp_min(1e-8).rsqrt()


class PriorityScorer(nn.Module):
    def __init__(self, variant="fused_readout"):
        super().__init__()
        self.variant = variant
        self.encoder = GCN(1, 128, 5, 1)
        if variant == "degree_residual":
            self.priority_head = nn.Sequential(nn.Linear(128+5+1, 24), nn.SiLU(), nn.Linear(24, 1))
            output = self.priority_head[-1]
        else:
            self.priority_head = nn.Linear(128 + 5, 1)
            output = self.priority_head
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    @classmethod
    def from_fused(cls, weights, device, variant="fused_readout"):
        model = cls(variant).to(device)
        model.encoder.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
        return model

    @classmethod
    def load(cls, checkpoint, device="cuda:0"):
        saved = torch.load(Path(checkpoint), map_location=device, weights_only=True)
        model = cls(saved.get("variant", "fused_readout")).to(device)
        model.load_state_dict(saved["model"], strict=True)
        return model.eval()

    def prepare(self, crow, col):
        n = crow.numel() - 1
        dtype = torch.float16 if crow.is_cuda else torch.float32
        adjacency = torch.sparse_csr_tensor(crow, col, torch.ones_like(col, dtype=dtype),
                                            size=(n, n)).to_dense()
        degree = crow.diff().float()
        d = degree.clamp_min(1)
        z = standardize(degree)
        neighbor_mean = (adjacency @ z.to(dtype)).float() / d
        neighbor_second = (adjacency @ z.square().to(dtype)).float() / d
        triangles = ((adjacency @ adjacency) * adjacency).sum(dim=1, dtype=torch.float32)
        clustering = triangles / (d * (degree - 1).clamp_min(1))
        structure = torch.stack((z, standardize(neighbor_mean),
                                 standardize((neighbor_second-neighbor_mean.square()).clamp_min(0).sqrt()),
                                 standardize(clustering), degree / degree.mean().clamp_min(1)), dim=1)
        inverse = (degree + 1).rsqrt().to(dtype)
        adjacency.fill_diagonal_(1)
        normalized = adjacency * inverse[:, None] * inverse[None, :]
        return normalized, structure

    def forward(self, normalized, structure):
        occupancy = (-structure[:, :1]).sigmoid()
        with torch.autocast(device_type=occupancy.device.type, dtype=torch.bfloat16, enabled=occupancy.is_cuda):
            hidden = self.encoder.initial_project(occupancy)
            for conv, linear in zip(self.encoder.convs, self.encoder.linears):
                hidden = hidden + F.relu(conv(hidden, normalized) + linear(hidden))
            original = self.encoder.final_project(hidden)[:, 0].float()
            if self.variant == "degree_residual":
                channels = standardize(hidden.float()).clamp(-5, 5)
                extra = self.priority_head(torch.cat((channels, structure, standardize(original)[:, None]), dim=1))
                return -structure[:, 0] + 0.35 * torch.tanh(extra[:, 0].float())
            extra = self.priority_head(torch.cat((F.layer_norm(hidden, (128,)).float(), structure), dim=1))
        return standardize(original) + extra[:, 0].float()

    @torch.inference_mode()
    def score(self, crow, col):
        return self(*self.prepare(crow, col)).contiguous()
