"""Save the initial statistics scorers for the fixed-work learning control."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'models'))
from mis_v3_value import PAVEV3, STATS_DIM


def main():
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        model = PAVEV3('stats')
        result = ROOT / 'outputs' / 'v3' / f'stats_untrained_h1_seed{seed}'
        checkpoints = result / 'checkpoints'
        checkpoints.mkdir(parents=True, exist_ok=True)
        torch.save(dict(mode='stats', stats_dim=STATS_DIM, model_state=model.state_dict(),
                        seed=seed, epoch=0, strongest_rule='degree'), checkpoints / 'best.pt')
        metadata = dict(version='v3', mode='untrained_control', status='completed',
                        run=result.name, model='stats', train_seed=seed, epoch=0,
                        strongest_rule='degree', params=sum(p.numel() for p in model.parameters()),
                        finished_at=datetime.now(timezone.utc).isoformat())
        (result / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
        print(f'{result.name}: {metadata["params"]} parameters, epoch 0', flush=True)


if __name__ == '__main__':
    main()
