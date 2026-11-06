#!/usr/bin/env python3
"""Measure actual GPU memory and throughput for graph-batched RLNN updates."""
import argparse,json,pickle,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'models'))
import torch
from main import build_parser,set_seed
from solvers import RLNN,erdoes_graph_batch_loss
from utils import build_batch
from problems.mis import Dataset

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--graph-batch',type=int,required=True)
parser.add_argument('--allocator-label',default='native')
args=parser.parse_args();G=args.graph_batch
folder=ROOT/'outputs/v000_rlnn_baseline'/f'graph_batch_probe_{G}_{args.allocator_label}'
folder.mkdir(parents=True,exist_ok=False)
config=build_parser().parse_args(['--method','RLNN','--loss','erdoes','--problem','mis','--num_l','5','--num_h','128','--num_t','100','--num_k','20','--num_tp','50','--num_kp','10','--num_d','5','--lambd','0.5','--lr','0.0001','--batch_size','1024','--graph_batch_size',str(G),'--train_path',str(ROOT/'models/data/MIS/RB_small_train'),'--valid_path',str(ROOT/'models/data/MIS/RB_small_val'),'--num_workers','0','--do_train'])
set_seed(0);solver=RLNN(config);solver.prepare()
dataset=Dataset(config.train_path,torch.float32)
sizes=[]
for i,path in enumerate(dataset.files):
    with open(path,'rb') as stream:g=pickle.load(stream)
    sizes.append((g.number_of_nodes(),g.number_of_edges(),i))
worst=[i for n,e,i in sorted(sizes,reverse=True)[:G]]
indices=[list(range(G)),list(range(G,2*G)),worst]
records=[]
for batch_index,selected in enumerate(indices):
    graphs=[dataset[i].to(solver.accelerator.device) for i in selected]
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    g=build_batch(graphs)
    inputs=solver.generate_data(g)
    torch.cuda.synchronize();generated=time.perf_counter()
    A=torch.sparse_coo_tensor(g.edge_index,g.edge_weight,(g.num_nodes,g.num_nodes)).to_sparse_csr()
    prob=torch.sigmoid(solver.model(inputs.unsqueeze(-1),g.edge_index).squeeze(-1))
    loss=erdoes_graph_batch_loss(solver.problem,A,g.b,inputs,prob,g.batch,len(graphs),5,0.5)
    solver.optimizer.zero_grad();solver.accelerator.backward(loss);solver.optimizer.step()
    torch.cuda.synchronize();end=time.perf_counter()
    smi=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')
    records.append({'graphs':G,'nodes':g.num_nodes,'states_per_graph':inputs.shape[1],'warmup':batch_index==0,'largest_graph_stress':batch_index==2,'generation_s':generated-start,'update_s':end-generated,'seconds_per_graph':(end-start)/G,'allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'current_reserved_mib':torch.cuda.memory_reserved()/1024**2,'allocation_retries':torch.cuda.memory_stats()['num_alloc_retries'],'device_used_mib':int(smi[0]),'device_total_mib':int(smi[1]),'loss':loss.item()})
    print(json.dumps(records[-1]),flush=True)
    del graphs,inputs,g,A,prob,loss
summary={'graph_batch_size':G,'configured_state_batch':1024,'effective_states_per_update':G*1000,'measured_graph_time_s':records[1]['seconds_per_graph'],'largest_graph_time_s':records[2]['seconds_per_graph'],'peak_device_used_mib':max(r['device_used_mib'] for r in records),'device_total_mib':records[0]['device_total_mib'],'peak_allocated_mib':max(r['allocated_mib'] for r in records),'peak_reserved_mib':max(r['reserved_mib'] for r in records)}
summary['peak_device_percent']=summary['peak_device_used_mib']/summary['device_total_mib']*100
(folder/'probe.json').write_text(json.dumps({'config':vars(config),'summary':summary,'generation_mode':'block_graph','allocator_backend':torch.cuda.memory.get_allocator_backend(),'batches':records},indent=2)+'\n')
print(json.dumps(summary,indent=2))
