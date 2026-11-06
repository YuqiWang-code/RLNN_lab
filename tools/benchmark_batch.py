#!/usr/bin/env python3
"""Measure RLNN graph throughput and memory for selected state batches."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'models'))
import torch
from main import build_parser,set_seed
from solvers import RLNN

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch-size',type=int,required=True)
parser.add_argument('--graphs',type=int,default=8)
args=parser.parse_args()
run=ROOT/'outputs/v000_rlnn_baseline'/f'batch_benchmark_{args.batch_size}'
run.mkdir(parents=True,exist_ok=False)
config=build_parser().parse_args(['--method','RLNN','--problem','mis','--loss','erdoes','--num_l','5','--num_h','128','--num_t','100','--num_k','20','--num_tp','50','--num_kp','10','--num_d','5','--beta','1.02','--lambd','0.5','--lr','0.0001','--batch_size',str(args.batch_size),'--epochs','1','--train_samples',str(args.graphs),'--valid_samples','1','--num_workers','0','--train_path',str(ROOT/'models/data/MIS/RB_small_train'),'--valid_path',str(ROOT/'models/data/MIS/RB_small_val'),'--do_train'])
set_seed(0)
solver=RLNN(config); solver.prepare()
generate=solver.generate_data; evaluate=solver.evaluate; step=solver.optimizer.step
records=[]; active=None; steps=0

def close_graph():
    global active
    if active is None:return
    torch.cuda.synchronize()
    active.update(total_s=time.perf_counter()-active.pop('started'),updates=steps-active.pop('start_steps'),peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    active['update_s']=active['total_s']-active['generation_s']
    records.append(active); active=None

def measured_generate(graph):
    global active
    close_graph(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start=time.perf_counter()
    inputs=generate(graph)
    torch.cuda.synchronize()
    active={'started':start,'start_steps':steps,'nodes':graph.num_nodes,'states':inputs.shape[1],'generation_s':time.perf_counter()-start}
    return inputs

def measured_step(*a,**kw):
    global steps
    value=step(*a,**kw); steps+=1; return value

def measured_evaluate(*a,**kw):
    close_graph(); return evaluate(*a,**kw)

solver.generate_data=measured_generate;solver.optimizer.step=measured_step;solver.evaluate=measured_evaluate
started=time.perf_counter()
solver.train(str(run))
measured=records[1:]
summary={key:sum(record[key] for record in measured)/len(measured) for key in ['total_s','generation_s','update_s','updates']}
summary.update(peak_allocated_mib=max(record['peak_allocated_mib'] for record in records),peak_reserved_mib=max(record['peak_reserved_mib'] for record in records),batch_size=args.batch_size,effective_batch=min(args.batch_size,1000),measured_graphs=len(measured),warmup_graphs=1)
report={'started_at':datetime.now(timezone.utc).isoformat(),'config':vars(config),'GPU':'1','torch':torch.__version__,'summary':summary,'graphs':records,'wall_time_s':time.perf_counter()-started}
(run/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
