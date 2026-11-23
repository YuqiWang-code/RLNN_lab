"""Compare four-particle and all-particle A.3 decoding at 200 ms."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, median

from compare_mis_runs import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    summary = compare(args.run_dir / 'full', args.run_dir / 'decode_all')
    expected_keys = {(f'idx_{i}', str(seed)) for i in range(1000) for seed in (1017, 1019, 1021)}
    for arm, metrics, width in [('full', 'candidate', 4), ('decode_all', 'reference', 200)]:
        rows = []
        for file in sorted((args.run_dir / arm).glob('**/v11_rows.csv')):
            with file.open(encoding='utf-8', newline='') as stream:
                rows.extend(csv.DictReader(stream))
        if len(rows) != 3000 or {(r['graph_id'], r['seed']) for r in rows} != expected_keys:
            raise ValueError(f'{arm}: incomplete set of matched graph--seed pairs')
        expected = {'decode_particles': str(width), 'num_k': '200', 'dense': '1', 'dtype': 'fp16',
                    'compile_model': '1', 'search_backend': 'captured', 'decode_schedule': 'midpoint',
                    'no_pave': '1', 'skip_incumbent': '1', 'fixed_steps': '0', 'budget': '0.2'}
        if any(any(row[k] != v for k, v in expected.items()) for row in rows):
            raise ValueError(f'{arm}: run settings differ from the paired protocol')
        ready = [float(row['final_ready_ms']) for row in rows if row['final_ready_ms']]
        if any(t > 200 for t in ready) or any(not r['final_ready_ms'] and float(r['final_size']) for r in rows):
            raise ValueError(f'{arm}: incomplete or late update was scored')
        summary[metrics].update(decode_particles=width, median_final_ready_ms=median(ready) if ready else None,
                                mean_decode_ms=fmean(float(r['decode_ms']) for r in rows),
                                wall_overruns=sum(int(r['deadline_overrun']) for r in rows))
    (args.run_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
