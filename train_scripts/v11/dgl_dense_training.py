"""Evaluate the official GraphConv parameters with cached normalized adjacency."""
from types import MethodType

import torch


def dense_forward(self, graph, features):
    adjacency = getattr(graph, "normalized_adjacency", None)
    if adjacency is None:
        src, dst = graph.edges()
        source_norm = graph.out_degrees().clamp(min=1).to(features.dtype).rsqrt()
        target_norm = graph.in_degrees().clamp(min=1).to(features.dtype).rsqrt()
        adjacency = features.new_zeros((graph.num_nodes(), graph.num_nodes()))
        adjacency[dst, src] = source_norm[src] * target_norm[dst]
        graph.normalized_adjacency = adjacency

    hidden = features
    for index, layer in enumerate(self.layers):
        if index:
            hidden = self.dropout(hidden)
        hidden = adjacency @ hidden @ layer.weight + layer.bias
        if layer._activation is not None:
            hidden = layer._activation(hidden)
    return hidden


def use_dense_training(model):
    model.forward = MethodType(dense_forward, model)
    return model
