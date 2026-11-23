"""Paired deadline scores for the original and additional OnlineMIS seed panels."""
import argparse
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from compare_mis_runs import read_rows, summarize


def panel(fused_dir, pattern, online_file, seeds):
    fused = read_rows(fused_dir, pattern)
    online = read_rows(online_file)
    expected = {(f'idx_{i}', str(seed), '0') for i in range(1000) for seed in seeds}
    if fused.keys() != expected or online.keys() != expected:
        raise ValueError(f'Incomplete paired seed panel in {online_file}')
    for rows in (fused, online):
        if any((r[2] is None and r[0] != 0) or (r[2] is not None and r[2] > 200) for r in rows.values()):
            raise ValueError('Deadline score does not match availability')
    return fused, online


def summarize_panel(fused, online):
    diffs = {key: fused[key][0] - online[key][0] for key in fused}
    seeds = sorted({key[1] for key in diffs}, key=int)
    graph_diffs = np.array([fmean(diffs[f'idx_{i}', seed, '0'] for seed in seeds) for i in range(1000)])
    rng = np.random.default_rng(42)
    boot = np.concatenate([graph_diffs[rng.integers(0, 1000, size=(500, 1000))].mean(axis=1)
                           for _ in range(10)])
    together = [diffs[k] for k in diffs if fused[k][2] is not None and online[k][2] is not None]
    return {
        'graphs': len(graph_diffs), 'seeds': list(map(int, seeds)),
        'fused': summarize(fused, 200), 'online': summarize(online, 200),
        'mean_paired_size_delta_fused_minus_online': float(graph_diffs.mean()),
        'ci95': np.quantile(boot, [0.025, 0.975]).tolist(),
        'wins': sum(d > 0 for d in diffs.values()), 'ties': sum(d == 0 for d in diffs.values()),
        'losses': sum(d < 0 for d in diffs.values()),
        'both_delivered_count': len(together), 'mean_delta_when_both_delivered': fmean(together),
        'per_seed': {s: {tag: summarize({k: row for k, row in table.items() if k[1] == s}, 200)
                         for tag, table in [('fused', fused), ('online', online)]} for s in seeds},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel_dir', type=Path)
    args = parser.parse_args()
    old = panel(Path('outputs/v11/fused_pave_02s_deadline'), 'pave_off_*/v11_rows.csv',
                Path('outputs/v11/reference_deadline_20261119/kamis_online/per_graph.csv'), (1017, 1019, 1021))
    new = panel(args.panel_dir / 'fused', '**/v11_rows.csv',
                args.panel_dir / 'kamis_online/per_graph.csv', (1023, 1025, 1027))
    results = dict(original=summarize_panel(*old), additional=summarize_panel(*new),
                   pooled=summarize_panel({**old[0], **new[0]}, {**old[1], **new[1]}))
    (args.panel_dir / 'summary.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
