"""A global token for the flip policy.

The frozen backbone is a five layer GCN, so a node sees at most five hops of a
231 node graph -- on RB instances that is a small fraction of the instance, and
the plateau study showed the resulting policy walks into a local optimum whose
one-flip and two-swap neighbourhoods are both empty.  The cheap fix is to give
every node the same *global* summary of the current state: mean and max of the
layer activations, projected back into the hidden width and added to every node
before the non-linearity.

Only the forward pass changes, so the move set, the loss and the training recipe
are untouched; the point of the experiment is to isolate "does global context
move the plateau?" from every other change.

Note on padding: the pooled summary is taken over real nodes only, so this
backbone is evaluated with the per-graph capture engine (``CapturedSearch``),
not with the amortised padded engine -- padding would fold the zero rows into
the mean.
"""

import torch
from torch import nn

from models import GCNConv, edge2adj


class GlobalTokenGCN(nn.Module):
    """``models.models.GCN`` plus a pooled graph token at every layer."""

    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.5):
        super(GlobalTokenGCN, self).__init__()
        self.initial_project = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.tokens = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.linears.append(nn.Linear(hidden_channels, hidden_channels))
            self.tokens.append(nn.Linear(2 * hidden_channels, hidden_channels))
        self.final_project = nn.Linear(hidden_channels, out_channels)
        self.activation = nn.ReLU()

    def reset_parameters(self):
        self.initial_project.reset_parameters()
        for conv, linear, token in zip(self.convs, self.linears, self.tokens):
            conv.reset_parameters()
            linear.reset_parameters()
            token.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, x, edge_index, adj=None):
        if adj is None:
            adj = edge2adj(edge_index, x.shape[0])
        return self.project_hidden(self.initial_project(x), adj)

    def project_hidden(self, x, adj):
        for conv, linear, token in zip(self.convs, self.linears, self.tokens):
            neighbor = conv(x, adj)
            residual = linear(x)
            pooled = torch.cat((x.mean(dim=0), x.amax(dim=0)), dim=-1)
            broadcast = token(pooled).unsqueeze(0)
            x = self.activation(neighbor + residual + broadcast) + x
        return self.final_project(x)
