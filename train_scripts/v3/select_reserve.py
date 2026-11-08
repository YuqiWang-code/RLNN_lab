"""Choose one shared postprocessing reserve using validation graphs only."""
import json
from pathlib import Path
from statistics import mean

root=Path(__file__).resolve().parents[2]/'outputs/v3'
measurements=[]
for reserve in ('0.02','0.05','0.10'):
    runs=[json.loads((root/f'reserve_{post}_{reserve}'/post/'run.json').read_text()) for post in ('pave','random')]
    measurements.append(dict(reserve_s=float(reserve),mean_size=mean(run['mean_size'] for run in runs),
        deadline_overruns=sum(run['deadline_overruns'] for run in runs),
        mean_time_s=mean(run['mean_solve_time_s'] for run in runs)))
selected=max(measurements,key=lambda item:(-item['deadline_overruns'],item['mean_size'],-abs(item['reserve_s']-.05)))
result=dict(version='v3',split='validation',graphs=100,backbone_seed=0,reserve_s=selected['reserve_s'],
            candidates=measurements,criterion='No deadline overruns, then mean quality; prefer 50 ms on ties')
(root/'budget_selection.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
