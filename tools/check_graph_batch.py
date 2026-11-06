#!/usr/bin/env python3
"""Check graph-batch loss and gradient against independent graph averages."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'models'))
import torch
from torch_geometric.data import Data
from models import GCN
from problems.mis import Problem
from solvers import erdoes_graph_batch_loss
from utils import build_batch

torch.manual_seed(0)
def graph(n,edges):
    e=torch.tensor(edges,dtype=torch.long,device='cuda').T
    e=torch.cat([e,e.flip(0)],dim=1)
    return Data(edge_index=e,edge_weight=torch.ones(e.shape[1],device='cuda')/2,
                x=torch.ones(n,1,device='cuda'),b=-torch.ones(n,1,device='cuda'),num_nodes=n)
graphs=[graph(3,[(0,1),(1,2)]),graph(5,[(0,1),(0,2),(0,3),(0,4)])]
inputs=[torch.randint(0,2,(g.num_nodes,3),device='cuda').float() for g in graphs]
model=GCN(1,128,5,1).cuda();problem=Problem(1.02)
losses=[];outputs=[]
for g,x in zip(graphs,inputs):
    p=torch.sigmoid(model(x.unsqueeze(-1),g.edge_index).squeeze(-1));outputs.append(p)
    A=torch.sparse_coo_tensor(g.edge_index,g.edge_weight,(g.num_nodes,g.num_nodes)).to_sparse_csr()
    energy,_=problem.energy_func(A,g.b,(1-p)*x+p*(1-x),False)
    losses.append(energy.mean()+0.5*((p.sum(0)-5)**2).mean())
expected=torch.stack(losses).mean()
grad_expected=torch.autograd.grad(expected,tuple(model.parameters()))
g=build_batch(graphs);x=torch.cat(inputs)
p=torch.sigmoid(model(x.unsqueeze(-1),g.edge_index).squeeze(-1))
A=torch.sparse_coo_tensor(g.edge_index,g.edge_weight,(g.num_nodes,g.num_nodes)).to_sparse_csr()
actual=erdoes_graph_batch_loss(problem,A,g.b,x,p,g.batch,2,5,0.5)
grad_actual=torch.autograd.grad(actual,tuple(model.parameters()))
assert torch.allclose(p,torch.cat(outputs),atol=1e-5,rtol=1e-5)
assert torch.allclose(actual,expected,atol=1e-5,rtol=1e-5)
assert all(torch.allclose(a,b,atol=2e-5,rtol=2e-4) for a,b in zip(grad_actual,grad_expected))
print('GCN block graph forward, per-graph mean loss and parameter gradients: passed')
