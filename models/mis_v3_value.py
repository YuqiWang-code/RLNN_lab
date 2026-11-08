"""Statewise action scoring for legal MIS plateau search."""
from pathlib import Path

import torch
from torch import nn


STATS_DIM = 24
MODES = ("stats", "message", "frequency")


def build_state(graph, selected, actions, remaining=1.0, *, include_nodes=False):
    """Encode one candidate pool without executing its continuation."""
    state = {
        "stats": torch.as_tensor(graph.stats_features(selected, actions, remaining=remaining),
                                 dtype=torch.float32).reshape(-1, STATS_DIM),
    }
    if include_nodes:
        state.update(
            node_features=torch.as_tensor(graph.node_features(selected), dtype=torch.float32),
            edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long),
            removed=[torch.tensor(action.removed, dtype=torch.long) for action in actions],
            added=[torch.tensor(action.added, dtype=torch.long) for action in actions],
            affected=[torch.tensor(graph.affected_nodes(action), dtype=torch.long) for action in actions],
        )
    return state


def move_state(state, device, mode="stats"):
    """Move the tensors used by this scorer; retain state metadata."""
    result = dict(state)
    result["stats"] = torch.as_tensor(state["stats"], dtype=torch.float32, device=device)
    for name in ("values", "_pairs"):
        if name in state:
            result[name] = torch.as_tensor(state[name], device=device)
    if mode != "stats":
        result["node_features"] = torch.as_tensor(state["node_features"], dtype=torch.float32, device=device)
        result["edge_index"] = torch.as_tensor(state["edge_index"], dtype=torch.long, device=device)
        for name in ("removed", "added", "affected"):
            result[name] = [torch.as_tensor(nodes, dtype=torch.long, device=device) for nodes in state[name]]
    return result


def neighbor_mean(features, edge_index):
    row, col = edge_index
    total = features.new_zeros(features.shape)
    total.index_add_(0, row, features[col])
    degree = torch.bincount(row, minlength=len(features)).to(features.dtype).unsqueeze(1)
    return torch.where(degree > 0, total / degree.clamp_min(1), features)


def pool_nodes(features, groups, *, maximum=False):
    """Pool variable-size node sets; empty affected regions have zero encoding."""
    result = features.new_zeros((len(groups), features.shape[1]))
    if not groups or not any(len(nodes) for nodes in groups):
        return result
    lengths = torch.tensor([len(nodes) for nodes in groups], device=features.device)
    owners = torch.repeat_interleave(torch.arange(len(groups), device=features.device), lengths)
    indices = torch.cat(groups)
    if maximum:
        result.scatter_reduce_(0, owners.unsqueeze(1).expand(-1, features.shape[1]),
                               features[indices], reduce="amax", include_self=False)
    else:
        result.index_add_(0, owners, features[indices])
        result = result / lengths.clamp_min(1).unsqueeze(1)
    return result


def node_branch():
    return nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())


class PAVEV3(nn.Module):
    def __init__(self, mode="stats", stats_dim=STATS_DIM):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown PAVE v3 mode: {mode}")
        self.mode = mode
        self.stats_dim = stats_dim
        if mode == "stats":
            self.scorer = nn.Sequential(nn.Linear(stats_dim, 64), nn.SiLU(),
                                        nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
        else:
            self.first = node_branch()
            self.second = node_branch()
            self.scorer = nn.Sequential(nn.Linear(4 * 64 + stats_dim, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, state):
        device = next(self.parameters()).device
        stats = torch.as_tensor(state["stats"], dtype=torch.float32, device=device).reshape(-1, self.stats_dim)
        if len(stats) == 0:
            return stats.new_empty(0)
        if self.mode == "stats":
            return self.scorer(stats).squeeze(-1)
        state = move_state(state, device, self.mode)
        features = state["node_features"]
        neighbors = neighbor_mean(features, state["edge_index"])
        if self.mode == "frequency":
            first = (features + neighbors) * 0.5
            second = features - first
        else:
            first, second = features, neighbors
        nodes = torch.cat((self.first(first), self.second(second)), dim=-1)
        action = torch.cat((pool_nodes(nodes, state["removed"]),
                            pool_nodes(nodes, state["added"]),
                            pool_nodes(nodes, state["affected"]),
                            pool_nodes(nodes, state["affected"], maximum=True), stats), dim=-1)
        return self.scorer(action).squeeze(-1)

    @torch.no_grad()
    def choose(self, graph, selected, actions, remaining=1.0):
        if len(actions) == 1:
            return actions[0]
        if not actions:
            return None
        state = build_state(graph, selected, actions, remaining, include_nodes=self.mode != "stats")
        return actions[int(self(state).argmax())]


def load_pave_v3(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = PAVEV3(checkpoint["mode"], checkpoint.get("stats_dim", STATS_DIM))
    model.load_state_dict(checkpoint["model_state"])
    model.strongest_rule = checkpoint["strongest_rule"]
    return model.to(device).eval()
