"""A symmetric local feature encoder for PAVE exchange ranking."""
import torch
from torch import nn


class PAVE(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(9, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, features):
        return super().forward(features).squeeze(-1)

    def choose(self, graph, selected, actions):
        if len(actions) == 1:
            return actions[0]
        features = torch.tensor([graph.features(selected, action) for action in actions])
        return actions[int(self(features).argmax())]
