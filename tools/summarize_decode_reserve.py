"""Compare leading decoding and a reserved search tail with their matched saved controls."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_mis_runs import compare, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    seeds = (1017, 1019, 1021)
    expected = {(f'idx_{i}', str(s), '0') for i in range(1000) for s in seeds}
    prelude_stats = {}
    for arm in ('leading', 'holdout'):
        rows = read_rows(args.run_dir / arm)
        if rows.keys() != expected or any((r[2] is None and r[0] != 0) or
                                           (r[2] is not None and r[2] > 200) for r in rows.values()):
            raise ValueError(f'Unexpected deadline scores for {arm}')
        preludes = []
        for s in seeds:
            path = args.run_dir / arm / f'batch_{s}/seed_{s}/padded/padded_ref0.json'
            meta = json.loads(path.read_text(encoding='utf-8'))
            required = dict(seeds=[s], budgets=[0.2], graphs=1000, compile_model=True, num_k=200,
                            dense=1, dtype='fp16', decode_top=4, decode_schedule='midpoint', pave=False,
                            leading_decode='on' if arm == 'leading' else 'off',
                            reserve_ms=40.0 if arm == 'holdout' else 0.0,
                            checkpoint_dir='outputs/v2/none_seed0/checkpoints')
            if any(meta[k] != v for k, v in required.items()):
                raise ValueError(f'Configuration mismatch in {path}')
            with path.with_suffix('.csv').open(encoding='utf-8', newline='') as stream:
                raw = list(csv.DictReader(stream))
            if len(raw) != 1000 or {r['graph_id'] for r in raw} != {f'idx_{i}' for i in range(1000)}:
                raise ValueError(f'Incomplete prelude timings for seed {s}')
            preludes.extend(1000 * float(r['prelude_s']) for r in raw)
        prelude_stats[arm] = dict(p50_ms=float(np.median(preludes)), p95_ms=float(np.quantile(preludes, 0.95)),
                                  max_ms=max(preludes))
    full = Path('outputs/v11/paper_ablations/decode_width_20261120/full')
    old = Path('outputs/v11/fused_pave_02s_deadline')
    hold = args.run_dir / 'holdout'
    result = dict(leading_minus_full=compare(args.run_dir / 'leading', full),
                  pave_minus_holdout=compare(old, hold, candidate_glob='pave_on_*/v11_rows.csv'),
                  full_minus_holdout=compare(old, hold, candidate_glob='pave_off_*/v11_rows.csv'),
                  prelude=prelude_stats)
    (args.run_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
