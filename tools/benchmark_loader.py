#!/usr/bin/env python3
"""Benchmark real RLNN epochs including graph loading and device transfer."""
from pathlib import Path
import argparse,json,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'models'))
import torch
from main import build_parser,set_seed
from solvers import RLNN

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--format',choices=['raw','tensor'],required=True)
parser.add_argument('--workers',type=int,required=True)
args=parser.parse_args()
name=f'loader_benchmark_{args.format}_w{args.workers}'
folder=ROOT/'outputs/v000_rlnn_baseline'/name;folder.mkdir(parents=True,exist_ok=False)
base=ROOT/'models/data'/('MIS' if args.format=='raw' else 'MIS_tensors')
config=build_parser().parse_args(['--method','RLNN','--loss','erdoes','--problem','mis','--num_l','5','--num_h','128','--num_k','20','--num_t','100','--num_tp','50','--num_kp','10','--num_d','5','--lr','0.0001','--lambd','0.5','--epochs','2','--batch_size','1024','--graph_batch_size','13','--train_path',str(base/'RB_small_train'),'--valid_path',str(base/'RB_small_val'),'--train_samples','104','--valid_samples','2','--num_workers',str(args.workers),'--do_train'])
set_seed(0);solver=RLNN(config);solver.prepare()
monitor=subprocess.Popen(['nvidia-smi','--id=1','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits','-lms','200'],stdout=subprocess.PIPE,text=True)
samples=[]
def read_monitor():
    for line in monitor.stdout:
        values=line.strip().split(',')
        if len(values)==2:samples.append((time.time(),float(values[0]),float(values[1])))
thread=threading.Thread(target=read_monitor,daemon=True);thread.start()
class TimedLoader:
    def __init__(self,loader):self.loader=loader;self.epochs=[]
    def __len__(self):return len(self.loader)
    def __iter__(self):
        started=time.perf_counter();utc_start=time.time();iterator=iter(self.loader);records=[]
        while True:
            before=time.perf_counter()
            try:graphs=next(iterator)
            except StopIteration:break
            torch.cuda.synchronize();ready=time.perf_counter()
            yield graphs
            torch.cuda.synchronize();end=time.perf_counter()
            records.append({'graphs':len(graphs),'input_wait_s':ready-before,'compute_s':end-ready})
        stopped=time.time();points=[row for row in samples if utc_start<=row[0]<=stopped]
        self.epochs.append({'train_wall_s':time.perf_counter()-started,'graphs':sum(r['graphs'] for r in records),'input_wait_s':sum(r['input_wait_s'] for r in records),'compute_s':sum(r['compute_s'] for r in records),'gpu_util_mean':sum(p[1] for p in points)/len(points) if points else None,'gpu_util_max':max((p[1] for p in points),default=None),'gpu_memory_max_mib':max((p[2] for p in points),default=None),'batches':records})
loader=TimedLoader(solver.train_loader);solver.train_loader=loader
try:solver.train(str(folder))
finally:
    monitor.terminate();thread.join(timeout=2)
report={'format':args.format,'workers':args.workers,'config':vars(config),'epochs':loader.epochs}
report['steady_seconds_per_graph']=loader.epochs[-1]['train_wall_s']/104
(folder/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:value for key,value in report.items() if key not in ['config','epochs']},indent=2))
print(json.dumps(loader.epochs[-1],indent=2))
