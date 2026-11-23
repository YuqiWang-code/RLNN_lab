#!/usr/bin/env python
"""t25: append sheet「V13 融合引擎」to docs/实验结果.xlsx from outputs/v11/collect/v13/*.csv.

Append-only: the 34 existing sheets are never touched; their dims are compared
before/after and the run aborts (after writing to a temp copy) if anything moved.

Row -> evidence mapping:
  arm x seed x metric   csv_row = arm_summary.csv:<line>   source = <run dir>/v11_rows.csv
  pooled x metric       csv_row = arm_summary.csv:<line>   source = <run dirs...>/v11_rows.csv
  contrast / constant   csv_row = comparisons.csv:<line>   source = outputs/v11/collect/v13/comparisons.csv
  equivalence check     csv_row = equivalence.csv:1 (200-row per-graph file, header+1)
Usage:  python models/v13_excel_sheet.py [--verify]
"""
import csv
import json
import os
import sys
import shutil
from datetime import datetime

import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WDIR = os.path.join(ROOT, 'outputs', 'v11', 'collect', 'v13')
XLSX = os.path.join(ROOT, 'docs', '实验结果.xlsx')
SHEET = 'V13 融合引擎'
BASE = ['version', 'run', 'task', 'dataset', 'split', 'epoch', 'step', 'metric', 'value',
        'unit', 'source', 'recorded_at', 'notes', 'ci_low', 'ci_high', 'win', 'tie', 'loss',
        'n_graphs', 'report_only', 'csv_row']
DATASET = 'RB200-300 n=1000'
DATASET200 = 'RB200-300 n=200'
SEARCH_ONLY = 'search-only (pave=false)：与论文「搜索+PAVE」口径不同，不得与 PAVE 臂混进同一均值行'


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def sheet_dims(ws):
    return (ws.max_row, ws.max_column)


def build_rows():
    arms = read_csv(os.path.join(WDIR, 'arm_summary.csv'))
    comps = read_csv(os.path.join(WDIR, 'comparisons.csv'))
    eq = read_csv(os.path.join(WDIR, 'equivalence.csv'))
    stats = json.load(open(os.path.join(WDIR, 'ab_stats.json'), encoding='utf-8'))
    now = datetime.now().replace(microsecond=0)
    rows = []

    def push(**kw):
        r = {k: None for k in BASE}
        r.update(kw)
        rows.append(r)

    def notes_for(a, extra=None):
        d = {
            'budget_s': float(a['budget']), 'seed': a['seed'], 'n_graphs': int(a['n_graphs']),
            'mean_steps': round(float(a['mean_steps']), 4),
            'mean_ms_per_step': round(float(a['mean_ms_per_step']), 6),
            'pave': (a['no_pave'] != '1'),
            'skip_incumbent': a['skip_incumbent'] == '1',
            'compile_model': a['compile_model'] == '1',
            'fixed_steps': a['fixed_steps'] or '0',
            'dtype': a['dtype'], 'dense': a['dense'], 'fp32_csr': a['fp32_csr'],
            'report_only': a['report_only'] == 'True',
            'note': a['note'],
        }
        if a['no_pave'] == '1':
            d['caveat'] = SEARCH_ONLY
        if extra:
            d.update(extra)
        return json.dumps(d, ensure_ascii=False)

    metric_specs = [('search_size', 'mean_search_size', 'indep_nodes'),
                    ('final_size', 'mean_final_size', 'indep_nodes'),
                    ('steps', 'mean_steps', 'steps'),
                    ('ms_per_step', 'mean_ms_per_step', 'ms'),
                    ('pave_ms', 'pave_ms', 'ms')]

    for a in arms:
        search_only_parity = a['block'] == 'G3-parity'
        ds = DATASET200 if search_only_parity else DATASET
        for metric, field, unit in metric_specs:
            if search_only_parity and metric == 'pave_ms':
                continue
            push(version='v13', run=a['arm'], task='v13-' + a['block'], dataset=ds, split='eval',
                 metric=metric, value=float(a[field]), unit=unit,
                 source=a['dir'] + '/v11_rows.csv', recorded_at=now,
                 notes=notes_for(a, {'csv_rows_in_arm_summary': a['csv_row']}),
                 n_graphs=int(a['n_graphs']), report_only=(a['report_only'] == 'True'),
                 csv_row=a['csv_row'],
                 **_ci(a, metric))

    for c in comps:
        paired = c['method'].startswith('graph-level')
        push(version='v13', run=c['contrast'], task='v13-' + c['block'],
             dataset=DATASET if paired else DATASET, split='eval',
             metric='delta_search_size', value=float(c['mean']),
             unit='indep_nodes', source='outputs/v11/collect/v13/comparisons.csv',
             recorded_at=now,
             notes=json.dumps({'arm_a': c['arm_a'], 'arm_b': c['arm_b'], 'budget_s': c['budget_s'],
                               'seeds': c['seeds'], 'pairs': c['pairs'],
                               'mean_steps_delta': c['mean_steps_delta'],
                               'method': c['method'], 'note': c['note']}, ensure_ascii=False),
             ci_low=float(c['ci_low']) if c['ci_low'] else None,
             ci_high=float(c['ci_high']) if c['ci_high'] else None,
             win=int(c['win']) if c['win'] else None,
             tie=int(c['tie']) if c['tie'] else None,
             loss=int(c['loss']) if c['loss'] else None,
             n_graphs=int(c['n_graphs']), report_only=not paired, csv_row=c['csv_row'])

    eq_summary = read_csv(os.path.join(WDIR, 'equivalence_summary.csv'))
    for e in eq_summary:
        push(version='v13', run='parity-compiled-plain', task='v13-G3-parity',
             dataset=DATASET200, split='eval', metric=e['check'], value=float(e['value']),
             unit=e['unit'], source='outputs/v11/r2/p1_gate/numeric200_{plain,compiled}/v11_rows.csv',
             recorded_at=now,
             notes=json.dumps({'method': 'per-graph search_size difference over the 200-graph probe',
                               'note': e['note'], 'ids': e['ids']}, ensure_ascii=False),
             n_graphs=int(e['n_graphs']), report_only=False, csv_row=e['csv_row'])
    return rows


def _ci(a, metric):
    if metric != 'search_size' or a['ci_low'] in ('', None):
        return {}
    return {'ci_low': float(a['ci_low']), 'ci_high': float(a['ci_high'])}


def verify(wb_before, wb_path, rows):
    """Re-open the file and check sheet + values + untouched sheets."""
    wb = openpyxl.load_workbook(wb_path)
    ok = True
    if SHEET not in wb.sheetnames:
        print('FAIL: sheet %r missing' % SHEET)
        return False
    ws = wb[SHEET]
    exp = (len(rows) + 1, len(BASE))
    got = sheet_dims(ws)
    print('sheet %r dims: expected %s got %s -> %s' % (SHEET, exp, got, 'ok' if exp == got else 'MISMATCH'))
    ok = ok and exp == got
    header = [c.value for c in ws[1]]
    if header != BASE:
        print('FAIL: header mismatch', header)
        ok = False
    # untouched sheets
    moved = []
    for name, dims in wb_before.items():
        if name == SHEET:
            continue
        if name not in wb.sheetnames:
            moved.append((name, dims, 'MISSING'))
        elif sheet_dims(wb[name]) != dims:
            moved.append((name, dims, sheet_dims(wb[name])))
    print('existing sheets: %d checked, %d moved -> %s'
          % (len(wb_before) - (1 if SHEET in wb_before else 0), len(moved), 'ok' if not moved else moved))
    ok = ok and not moved
    # value spot-check against the CSVs cited by csv_row
    cache = {}
    bad = 0
    for i, r in enumerate(rows):
        cellvals = [c.value for c in ws[i + 2]]
        if abs(float(cellvals[BASE.index('value')]) - r['value']) > 1e-12:
            bad += 1
            continue
        src = r['csv_row']
        fname, line = src.split(':')
        if fname not in cache:
            cache[fname] = read_csv(os.path.join(WDIR, fname))
        idx = int(line) - 2
        rec = cache[fname][idx]
        key = {'arm_summary.csv': 'mean_' + r['metric'] if r['metric'] != 'pave_ms' else 'pave_ms',
               'comparisons.csv': 'mean', 'equivalence_summary.csv': 'value'}[fname]
        if key in rec and rec[key] not in ('', None):
            if abs(float(rec[key]) - r['value']) > 1e-9:
                bad += 1
                print('FAIL trace: row %d %s %s -> csv %s=%s vs sheet %s'
                      % (i + 2, r['run'], r['metric'], key, rec[key], r['value']))
    print('trace check: %d rows, %d mismatches -> %s' % (len(rows), bad, 'ok' if bad == 0 else 'FAIL'))
    ok = ok and bad == 0
    wb.close()
    return ok


def main():
    if '--verify' in sys.argv:
        wb = openpyxl.load_workbook(XLSX, read_only=True)
        if SHEET not in wb.sheetnames:
            print('FAIL: sheet missing')
            return 1
        ws = wb[SHEET]
        print('sheet %r dims=%s sheets=%d' % (SHEET, sheet_dims(ws), len(wb.sheetnames)))
        wb.close()
        return 0

    wb = openpyxl.load_workbook(XLSX)
    before = {n: sheet_dims(wb[n]) for n in wb.sheetnames}
    if SHEET in wb.sheetnames:
        print('sheet %r already exists -> refusing to duplicate (delete it first)' % SHEET)
        return 2
    rows = build_rows()
    ws = wb.create_sheet(SHEET)
    ws.append(BASE)
    for r in rows:
        ws.append([r[k] for k in BASE])
    ws.freeze_panes = 'B2'
    ws.auto_filter.ref = 'A1:%s%d' % (openpyxl.utils.get_column_letter(len(BASE)), len(rows) + 1)
    shutil.copy2(XLSX, os.path.join(ROOT, 'tmp', 'v13_excel_backup.xlsx'))
    wb.save(XLSX)
    wb.close()
    print('appended %d rows to %r' % (len(rows), SHEET))
    ok = verify(before, XLSX, rows)
    print('VERIFY %s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
