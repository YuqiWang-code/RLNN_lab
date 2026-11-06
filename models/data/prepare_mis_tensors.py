#!/usr/bin/env python3
"""Convert MIS graphs once while preserving the original loader order."""
import argparse,json,sys
from pathlib import Path
from multiprocessing import Pool
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
torch.set_num_threads(1)
from problems.mis import Dataset

def convert(job):
    source,destination,idx=job
    dataset=Dataset(source,torch.float32)
    graph=dataset[idx]
    name=f'graph_{idx:06d}.pt'
    torch.save({'edge_index':graph.edge_index,'num_nodes':graph.num_nodes},Path(destination)/name)
    return {'source':Path(dataset.files[idx]).name,'tensor':name,'num_nodes':graph.num_nodes}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();args.destination.mkdir(parents=True,exist_ok=False)
    dataset=Dataset(str(args.source),torch.float32)
    jobs=[(str(args.source),str(args.destination),idx) for idx in range(len(dataset))]
    with Pool(args.workers) as pool: graphs=list(pool.imap(convert,jobs))
    manifest={'source_directory':str(args.source.resolve()),'count':len(graphs),'graphs':graphs}
    (args.destination/'tensor_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cached=Dataset(str(args.destination),torch.float32)
    for idx in [0,len(dataset)//2,len(dataset)-1]:
        a,b=dataset[idx],cached[idx]
        assert a.num_nodes==b.num_nodes
        for key in ['edge_index','edge_weight','x','b']:
            assert torch.equal(getattr(a,key),getattr(b,key)), (idx,key)
    print(f'{args.source}: {len(graphs)} graphs converted; edge order and tensor equality passed',flush=True)
if __name__=='__main__':main()
