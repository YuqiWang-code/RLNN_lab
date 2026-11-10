import torch
from torch_geometric.data import Data

def collate_fn(batch):
    return batch

def build_batch(graphs):
    num_nodes = 0  
    edge_index = []
    edge_weight = []
    x = []
    b = []
    graph_index = []
    graph_ptr = [0]
    for i, graph in enumerate(graphs):
        edge_index.append(graph.edge_index+num_nodes)
        edge_weight.append(graph.edge_weight)
        x.append(graph.x)
        b.append(graph.b)
        graph_index.append(torch.ones(graph.num_nodes, device=graph.x.device)*i)
        num_nodes += graph.num_nodes
        graph_ptr.append(num_nodes)
    x = torch.cat(x,0)
    b = torch.cat(b, 0)
    edge_index = torch.cat(edge_index,1)
    edge_weight = torch.cat(edge_weight,0)
    graph_index = torch.cat(graph_index,0).long()
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, b=b, num_nodes=num_nodes, batch=graph_index, num_graphs=len(graphs))
    data.graph_ptr = tuple(graph_ptr)
    if all(hasattr(graph, "idx") for graph in graphs):
        data.graph_ids = tuple(int(graph.idx) for graph in graphs)
    return data
