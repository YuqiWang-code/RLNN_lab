#!/usr/bin/env python3
"""Prepare an explicitly weights-only continuation from a best checkpoint."""
import argparse,json,re,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-run',required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();run=ROOT/'outputs/v000_rlnn_baseline'/args.source_run
history=[]
for line in (run/'train.log').read_text().splitlines():
    if 'Epoch:' not in line or 'valid_obj: None' in line:continue
    values=dict(re.findall(r'(Epoch|Iteration|loss|valid_obj|valid_time):\s*(\S+)',line))
    if values.get('valid_obj') in (None,'None'):continue
    history.append({key:int(value) if key in ['Epoch','Iteration'] else float(value) for key,value in values.items()})
best=min(history,key=lambda row:row['valid_obj']);epoch=best['Epoch']
source_meta=json.loads((run/'run.json').read_text())
weights=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=True)
state={'model':weights,'best_model':weights,'optimizer':None,'rng':None,'epoch':epoch,
       'best_eval':best['valid_obj'],'logger':best,'history':[row for row in history if row['Epoch']<=epoch],
       'resume_mode':'weights_only','source_run':str(run),'config':source_meta['config'],'device':source_meta['device']}
args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(state,args.output)
summary={'source_run':str(run),'source_completed_epochs':max(row['Epoch'] for row in history),
         'resume_epoch':epoch,'best_valid_obj':best['valid_obj'],'optimizer_reset':True,'random_state_reset':True}
args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
