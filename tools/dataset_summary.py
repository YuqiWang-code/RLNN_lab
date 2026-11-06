#!/usr/bin/env python3
"""Record RB split generation settings and observed graph sizes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import networkx as nx
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
report={'dataset':'RB200-300','source':'models/data/rb_generator.py + xu_util.py',
        'created_at':datetime.now(timezone.utc).isoformat(),'networkx':nx.__version__,
        'numpy':np.__version__,'requested_node_range':[200,300],'splits':{}}
for split,seed in [('train',0),('val',100000),('test',200000)]:
    folder=args.data_root/f'RB_small_{split}'
    files=sorted(folder.glob('*.gpickle'))
    if not files: raise ValueError(f'Empty split: {folder}')
    nodes=[]; edges=[]
    for path in files:
        with path.open('rb') as stream: graph=pickle.load(stream)
        nodes.append(graph.number_of_nodes()); edges.append(graph.number_of_edges())
        if not 200 <= nodes[-1] <= 300: raise ValueError(path)
    report['splits'][split]={'path':str(folder.resolve()),'count':len(files),'seed':seed,
        'per_graph_seed':'seed + idx','min_nodes':min(nodes),'max_nodes':max(nodes),
        'mean_nodes':sum(nodes)/len(nodes),'min_edges':min(edges),'max_edges':max(edges)}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
