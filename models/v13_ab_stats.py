#!/usr/bin/env python
"""t25: v13/v14 ablation aggregation -- CSV-only, EXPLICIT include prefixes.

Groups (contract):
  G1  0.2 s three-arm duel   C (lead on, PAVE on) / A (lead off, PAVE on) / B (lead off, PAVE off)
  G2  0.1 s PAVE on/off      A' / B'  (+ legacy C' = ladder 0.1 s, report-only)
  G3  fused engine           B / B+ (fused) / A0 (1 s)  + numeric equivalence probe
                              (L3 1 s fused = record-only)

Every number is recomputed from each run's own `v11_rows.csv` (per-graph rows,
repeats collapsed to their per-graph mean).  Outputs:

  outputs/v11/collect/v13/v13_rows_merged.csv   merged per-graph rows (all included runs)
  outputs/v11/collect/v13/arm_summary.csv       per arm x seed (+ pooled) means, with pave_ms
  outputs/v11/collect/v13/comparisons.csv       paired graph-level bootstrap contrasts
  outputs/v11/collect/v13/equivalence.csv       parity probe, per graph
  outputs/v11/collect/v13/ab_stats.json         everything + reconciliation vs t21 JSONs

`csv_row` in the workbook points at a line of arm_summary.csv / comparisons.csv;
`source` points at the raw run directory holding v11_rows.csv.
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, 'outputs', 'v11', 'collect', 'v13')

# ---- explicit include / exclude prefixes (printed, so the report can cite them) ----
INCLUDE_PREFIXES = [
    'outputs/v11/t7/a02_s1017_1000',            # G1 C seed 1017 (t7 read-only reuse)
    'outputs/v11/t7/a02_s1019_1000',            # G1 C seed 1019
    'outputs/v11/t7/a02_s1021_1000',            # G1 C seed 1021
    'outputs/v11/t7/v13_noprelude_02s_s*',      # G1 A
    'outputs/v11/t7/v13_nopave_02s_s*',         # G1 B
    'outputs/v11/t7/v13_fused_nopave_02s_s*',   # G3 B+
    'outputs/v11/t7/v13_noprelude_01s_s*',      # G2 A'
    'outputs/v11/t7/v13_nopave_01s_s*',         # G2 B'
    'outputs/v11/t7/v13_L3_1s_s*',              # G3 L3 (1 s fused, record-only)
    'outputs/v11/t7/ladder_s1017_1000',         # G2 C' (budget=0.1 slice only)
    'outputs/v11/gate0/a0_steps808_n1000',      # G3 A0 seed 1017
    'outputs/v11/t7/a0_s1019_steps808_1000',    # G3 A0 seed 1019
    'outputs/v11/t7/a0_s1021_steps808_1000',    # G3 A0 seed 1021
    'outputs/v11/r2/p1_gate/numeric200_plain',     # G3 parity: plain
    'outputs/v11/r2/p1_gate/numeric200_compiled',  # G3 parity: torch.compile
]
EXCLUDE_PREFIXES = [
    ('outputs/v11/t7/v13_L1_1s_s*', 'L1 = 1 s PAVE-on 基线，三方表用 A0 扮演同口径'),
    ('outputs/v11/t7/v13_L2_1s_s*', 'L2 = 1 s PAVE-off 非融合，只用于 L2-L1 PAVE 增益（t21 §7，本表不收）'),
    ('outputs/v11/t7/v13_fused_a0_1s_s1017', 'A0+融合 1 s 单点探针，属 P1 专项，不入三方表'),
    ('outputs/v11/t7/default_regression_s1017_n20', 'n=20 冒烟回归，非 1000 图评测'),
    ('outputs/v11/t7/smoke_v14b', 'v14b 冒烟'),
]

REPS = 5000
SEED = 20261118
TOL = 1e-9
DATASET = 'RB200-300 n=1000'
DATASET200 = 'RB200-300 n=200'

ARMS = [
    dict(block='G1-0.2s', arm='C(lead on,PAVE on)', budget=0.2, report_only=False,
         seed_dirs={1017: 'outputs/v11/t7/a02_s1017_1000',
                    1019: 'outputs/v11/t7/a02_s1019_1000',
                    1021: 'outputs/v11/t7/a02_s1021_1000'},
         note='现况（前导解码 on + PAVE on）；只读复用 t7 0.2 s 臂'),
    dict(block='G1-0.2s', arm='A(lead off,PAVE on)', budget=0.2, report_only=False,
         seed_dirs={s: 'outputs/v11/t7/v13_noprelude_02s_s%d' % s for s in (1017, 1019, 1021)},
         note='skip_incumbent=1（关前导解码），PAVE on'),
    dict(block='G1-0.2s', arm='B(lead off,PAVE off)', budget=0.2, report_only=False,
         seed_dirs={s: 'outputs/v11/t7/v13_nopave_02s_s%d' % s for s in (1017, 1019, 1021)},
         note='skip_incumbent=1 + no_pave=1 => search-only 解，口径与 PAVE 后处理不同'),
    dict(block='G3-fused', arm='B+(fused)', budget=0.2, report_only=False,
         seed_dirs={s: 'outputs/v11/t7/v13_fused_nopave_02s_s%d' % s for s in (1017, 1019, 1021)},
         note='B 臂 + compile_model=1（torch.compile 融合），search-only 口径'),
    dict(block='G2-0.1s', arm="A'(lead off,PAVE on)", budget=0.1, report_only=False,
         seed_dirs={s: 'outputs/v11/t7/v13_noprelude_01s_s%d' % s for s in (1017, 1019, 1021)},
         note='0.1 s，关前导解码，PAVE on'),
    dict(block='G2-0.1s', arm="B'(lead off,PAVE off)", budget=0.1, report_only=False,
         seed_dirs={s: 'outputs/v11/t7/v13_nopave_01s_s%d' % s for s in (1017, 1019, 1021)},
         note='0.1 s，关前导解码 + 关 PAVE，search-only 口径'),
    dict(block='G3-A0', arm='A0(1s,PAVE on)', budget=1.0, report_only=False,
         seed_dirs={1017: 'outputs/v11/gate0/a0_steps808_n1000',
                    1019: 'outputs/v11/t7/a0_s1019_steps808_1000',
                    1021: 'outputs/v11/t7/a0_s1021_steps808_1000'},
         note='v11 台账 A0（前导解码 on + PAVE on，fixed_steps=808）；seed 1017 为 gate0 只读产物'),
    dict(block='G3-fused-1s', arm='L3(1s,fused)', budget=1.0, report_only=True,
         seed_dirs={s: 'outputs/v11/t7/v13_L3_1s_s%d' % s for s in (1017, 1019, 1021)},
         note='1 s + 融合 + PAVE off（记录用；PAVE 在 1 s 档增益恒 0）'),
    dict(block='G2-legacy', arm="C'(legacy lead on)", budget=0.1, report_only=True,
         seed_dirs={1017: 'outputs/v11/t7/ladder_s1017_1000'},
         note='t7 ladder 0.1 s 旧协议（前导解码 on）；解释 A′ 第 1 个 +0.07 不是 PAVE 给的'),
    dict(block='G3-parity', arm='parity-plain(200g,130steps)', budget=1.0, report_only=True,
         seed_dirs={1017: 'outputs/v11/r2/p1_gate/numeric200_plain'},
         note='数值门基线：默认路径，fixed 130 步，200 图'),
    dict(block='G3-parity', arm='parity-compiled(200g,130steps)', budget=1.0, report_only=True,
         seed_dirs={1017: 'outputs/v11/r2/p1_gate/numeric200_compiled'},
         note='数值门实验臂：compile_model=1，其余同 plain'),
]

# paired contrasts: (label, block, budget, arm_a, arm_b, note)
CONTRASTS = [
    ('A - C', 'G1-0.2s', 0.2, 'A(lead off,PAVE on)', 'C(lead on,PAVE on)',
     '同一 0.2 s deadline、同一批 1000 图：关前导解码的收益'),
    ('B - A', 'G1-0.2s', 0.2, 'B(lead off,PAVE off)', 'A(lead off,PAVE on)',
     '关 PAVE 的收益 = 把 PAVE 预留还给搜索'),
    ('B - C', 'G1-0.2s', 0.2, 'B(lead off,PAVE off)', 'C(lead on,PAVE on)',
     '两处 additive 改动合计'),
    ("B' - A'", 'G2-0.1s', 0.1, "B'(lead off,PAVE off)", "A'(lead off,PAVE on)",
     '0.1 s 档 PAVE off - on'),
    ('B+ - B', 'G3-fused', 0.2, 'B+(fused)', 'B(lead off,PAVE off)',
     '融合引擎（torch.compile）在同预算下的收益'),
    ('B+ - A0', 'G3-fused', 0.2, 'B+(fused)', 'A0(1s,PAVE on)',
     '跨预算参照（0.2 s 融合 vs 1 s 现况；同图配对，非同预算等价性判据）'),
]
# unpaired constant references
CONSTANTS = [
    ('B - RLSA@0.2s', 'G1-0.2s', 'B(lead off,PAVE off)', 19.9620, 'RLSA 19.9620 @ 0.201 s（无逐图数据，未配对）'),
    ('B+ - RLSA@0.2s', 'G3-fused', 'B+(fused)', 19.9620, '同上'),
    ('A0 - RLSA@0.2s', 'G3-A0', 'A0(1s,PAVE on)', 19.9620, '同上（1 s vs 0.2 s，口径不同）'),
    ('B+ - A0', 'G3-fused', 'B+(fused)', None, '同图配对版本见 comparisons.csv；此处为 0.2 s 融合 vs 1 s 现况的池均值差'),
    ('L3 - RLSA@1s', 'G3-fused-1s', 'L3(1s,fused)', 20.084, 'RLSA 1 s 插值 20.084（无实测点）'),
    ('L3 - KaMIS@3s', 'G3-fused-1s', 'L3(1s,fused)', 20.1020, 'KaMIS 采用值 20.1020 @ 3.046 s'),
    ('A0 - KaMIS@3s', 'G3-A0', 'A0(1s,PAVE on)', 20.1020, '同上'),
]


def rel(p):
    return os.path.join(ROOT, p.replace('/', os.sep))


def load_cell(run_dir, budget):
    """graph_id -> per-graph row dict (repeats averaged)."""
    path = rel(run_dir + '/v11_rows.csv')
    if not os.path.exists(path):
        raise SystemExit('missing %s' % path)
    acc = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            gid = (r.get('graph_id') or '').strip()
            if not gid.startswith('graph'):
                continue
            b = float(r['budget'])
            if abs(b - budget) > 1e-9:
                continue
            acc.setdefault(gid, []).append(r)
    cell = {}
    for gid, rs in acc.items():
        def m(k):
            return sum(float(x[k]) for x in rs) / len(rs)
        cell[gid] = dict(
            search_size=m('search_size'), final_size=m('final_size'), steps=m('steps'),
            ms_per_step=m('ms_per_step'), wall_ms=m('wall_ms'), search_ms=m('search_ms'),
            pave_gain=m('pave_gain'),
            dtype=rs[0].get('dtype', ''), dense=rs[0].get('dense', ''),
            fp32_csr=rs[0].get('fp32_csr', ''), fixed_steps=rs[0].get('fixed_steps', ''),
            skip_incumbent=rs[0].get('skip_incumbent', ''), no_pave=rs[0].get('no_pave', ''),
            compile_model=rs[0].get('compile_model', ''), repeats=len(rs))
    if not cell:
        raise SystemExit('no rows after budget filter: %s budget=%s' % (run_dir, budget))
    return cell


def cell_stats(cell):
    n = len(cell)
    keys = ('search_size', 'final_size', 'steps', 'ms_per_step', 'wall_ms', 'search_ms', 'pave_gain')
    out = {'n_graphs': n}
    for k in keys:
        out['mean_' + k] = sum(v[k] for v in cell.values()) / n
    out['pave_ms'] = out['mean_wall_ms'] - out['mean_search_ms']
    first = next(iter(cell.values()))
    out.update({k: first[k] for k in ('dtype', 'dense', 'fp32_csr', 'fixed_steps',
                                      'skip_incumbent', 'no_pave', 'compile_model', 'repeats')})
    return out


def pooled_cell(cells):
    """per-graph mean across seeds (all cells share the 1000 graph ids)."""
    ids = [set(c) for c in cells]
    common = set.intersection(*ids)
    if len(common) != len(ids[0]):
        print('WARN graph ids differ across seeds: %d common of %d' % (len(common), len(ids[0])))
    numeric = [k for k, v in next(iter(cells[0].values())).items() if isinstance(v, (int, float))]
    out = {}
    for gid in common:
        row = {k: cells[0][gid][k] for k in cells[0][gid] if k not in numeric}
        row.update({k: sum(c[gid][k] for c in cells) / len(cells) for k in numeric})
        out[gid] = row
    return out


def boot_ci_cluster(delta_matrix):
    """delta_matrix: (n_graphs, n_seeds). Resample graphs (cluster), keep seeds."""
    arr = np.asarray(delta_matrix, dtype=float)
    n = arr.shape[0]
    rng = np.random.default_rng(SEED)
    means = np.empty(REPS)
    step = 500
    for start in range(0, REPS, step):
        k = min(step, REPS - start)
        idx = rng.integers(0, n, size=(k, n))
        means[start:start + k] = arr[idx].mean(axis=(1, 2))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def cell_ci(cell):
    """95% CI of the per-graph mean search_size (resample graphs)."""
    vals = np.array([v['search_size'] for v in cell.values()], dtype=float)
    rng = np.random.default_rng(SEED + 1)
    means = np.empty(REPS)
    step = 500
    for start in range(0, REPS, step):
        k = min(step, REPS - start)
        idx = rng.integers(0, vals.size, size=(k, vals.size))
        means[start:start + k] = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def contrast(name, block, cell_a, cell_b, budget, note):
    """cell_a/cell_b: dict seed -> collapsed cell."""
    seeds = sorted(set(cell_a) & set(cell_b))
    common = set.intersection(*[set(cell_a[s]) for s in seeds], *[set(cell_b[s]) for s in seeds])
    gids = sorted(common)
    mat = []
    wins = ties = losses = 0
    dsteps = []
    for gid in gids:
        row = []
        for s in seeds:
            d = cell_a[s][gid]['search_size'] - cell_b[s][gid]['search_size']
            row.append(d)
            if d > TOL:
                wins += 1
            elif d < -TOL:
                losses += 1
            else:
                ties += 1
        mat.append(row)
        dsteps.append(sum(cell_a[s][gid]['steps'] - cell_b[s][gid]['steps'] for s in seeds) / len(seeds))
    arr = np.asarray(mat, dtype=float)
    lo, hi = boot_ci_cluster(arr)
    return dict(contrast=name, block=block, budget_s=budget, seeds=','.join(str(s) for s in seeds),
                n_graphs=len(gids), pairs=int(arr.size),
                mean=float(arr.mean()), ci_low=lo, ci_high=hi,
                win=wins, tie=ties, loss=losses,
                mean_steps_delta=float(np.mean(dsteps)),
                method='graph-level cluster bootstrap (resample %d graphs, carry all %d seeds), %d reps, percentile 2.5/97.5' % (len(gids), len(seeds), REPS),
                note=note)


def drift(name, cell_a, cell_b, graph_ids=None):
    gids = sorted(set(cell_a) & set(cell_b)) if graph_ids is None else sorted(graph_ids)
    deltas = [(g, cell_a[g]['search_size'] - cell_b[g]['search_size']) for g in gids]
    ident = sum(1 for _, d in deltas if abs(d) <= TOL)
    ge2 = [(g, d) for g, d in deltas if abs(d) >= 2 - 1e-9]
    ge3 = [(g, d) for g, d in deltas if abs(d) >= 3 - 1e-9]
    return dict(name=name, n_graphs=len(deltas), identical=ident,
                within_one=sum(1 for _, d in deltas if abs(d) <= 1 + TOL),
                n_ge2=len(ge2), n_ge3=len(ge3),
                max_abs=max(abs(d) for _, d in deltas) if deltas else 0.0,
                mean=sum(d for _, d in deltas) / len(deltas) if deltas else 0.0,
                ge2_ids=[g for g, _ in ge2], ge3_ids=[g for g, _ in ge3],
                deltas=deltas)


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    print('INCLUDE prefixes (explicit):')
    for p in INCLUDE_PREFIXES:
        print('  +', p)
    print('EXCLUDE prefixes (explicit, with reason):')
    for p, why in EXCLUDE_PREFIXES:
        print('  -', p, '->', why)

    stats = {}
    merged = []
    per_arm_seed = {}
    for a in ARMS:
        per_arm_seed[a['arm']] = {}
        for s, d in sorted(a['seed_dirs'].items()):
            cell = load_cell(d, a['budget'])
            st = cell_stats(cell)
            st.update(arm=a['arm'], block=a['block'], budget=a['budget'], seed=s, dir=d,
                      report_only=a['report_only'], note=a['note'])
            st['ci_low'], st['ci_high'] = cell_ci(cell)
            # schema guard
            if st['dtype'] and st['dtype'] != 'fp16':
                print('WARN dtype %s in %s' % (st['dtype'], d))
            per_arm_seed[a['arm']][s] = cell
            stats['%s|%s' % (a['arm'], s)] = st
            for gid, v in cell.items():
                merged.append(dict(arm=a['arm'], block=a['block'], budget=a['budget'], seed=s,
                                   graph_id=gid, **{k: v[k] for k in
                                   ('search_size', 'final_size', 'steps', 'ms_per_step',
                                    'wall_ms', 'search_ms', 'pave_gain', 'dtype', 'dense',
                                    'fp32_csr', 'fixed_steps', 'skip_incumbent', 'no_pave',
                                    'compile_model')}))
            print('%-26s seed=%d graphs=%d search=%.4f steps=%.2f ms/step=%.4f pave_ms=%.2f'
                  % (a['arm'], s, st['n_graphs'], st['mean_search_size'], st['mean_steps'],
                     st['mean_ms_per_step'], st['pave_ms']))

    # ---- pooled rows ----
    pooled = {}
    pooled_rows = []
    for a in ARMS:
        seeds = sorted(per_arm_seed[a['arm']])
        if len(seeds) < 2:
            continue
        cells = [per_arm_seed[a['arm']][s] for s in seeds]
        pc = pooled_cell(cells)
        st = cell_stats(pc)
        st['ci_low'], st['ci_high'] = cell_ci(pc)
        st.update(arm=a['arm'], block=a['block'], budget=a['budget'], seed='pooled',
                  dir='+'.join(a['seed_dirs'][s] for s in seeds), report_only=a['report_only'],
                  note=a['note'])
        pooled[a['arm']] = pc
        pooled_rows.append(st)
        print('%-26s pooled  graphs=%d search=%.4f steps=%.2f ms/step=%.4f'
              % (a['arm'], st['n_graphs'], st['mean_search_size'], st['mean_steps'],
                 st['mean_ms_per_step']))

    # ---- arm_summary.csv (csv_row = its own line number) ----
    sum_fields = ['block', 'arm', 'budget', 'seed', 'n_graphs', 'mean_search_size', 'mean_final_size',
                  'mean_steps', 'mean_ms_per_step', 'mean_wall_ms', 'mean_search_ms', 'pave_ms', 'ci_low', 'ci_high',
                  'mean_pave_gain', 'dtype', 'dense', 'fp32_csr', 'fixed_steps', 'skip_incumbent',
                  'no_pave', 'compile_model', 'repeats', 'report_only', 'dir', 'note']
    sum_rows = []
    for a in ARMS:
        for s in sorted(per_arm_seed[a['arm']]):
            sum_rows.append(stats['%s|%s' % (a['arm'], s)])
    sum_rows.extend(pooled_rows)
    for i, r in enumerate(sum_rows):
        r['csv_row'] = 'arm_summary.csv:%d' % (i + 2)
    write_csv(os.path.join(OUTDIR, 'arm_summary.csv'), sum_rows, sum_fields + ['csv_row'])

    # ---- comparisons ----
    comp_rows = []
    for (label, block, budget, arm_a, arm_b, note) in CONTRASTS:
        ca = per_arm_seed[arm_a]
        cb = per_arm_seed[arm_b]
        r = contrast(label, block, ca, cb, budget, note)
        r['arm_a'] = arm_a
        r['arm_b'] = arm_b
        comp_rows.append(r)
        print('%-8s mean=%+.4f CI=[%+.4f,%+.4f] W/T/L=%d/%d/%d dsteps=%+.2f'
              % (label, r['mean'], r['ci_low'], r['ci_high'], r['win'], r['tie'], r['loss'],
                 r['mean_steps_delta']))

    for (label, block, arm, ref, note) in CONSTANTS:
        base = pooled.get(arm) or per_arm_seed[arm][sorted(per_arm_seed[arm])[0]]
        mean_val = sum(v['search_size'] for v in base.values()) / len(base)
        target = ref if ref is not None else pooled['A0(1s,PAVE on)']
        tval = (sum(v['search_size'] for v in target.values()) / len(target)) if isinstance(target, dict) else target
        comp_rows.append(dict(contrast=label, block=block, budget_s='',
                              arm_a=arm, arm_b=('A0(1s,PAVE on)' if ref is None else 'constant %.4f' % target),
                              seeds='n/a', n_graphs=len(base), pairs=0,
                              mean=mean_val - tval, ci_low='', ci_high='', win='', tie='', loss='',
                              mean_steps_delta='', method='unpaired constant difference',
                              note=note))
        print('%-16s unpaired diff=%+.4f' % (label, mean_val - tval))
    comp_fields = ['contrast', 'block', 'budget_s', 'arm_a', 'arm_b', 'seeds', 'n_graphs', 'pairs',
                   'mean', 'ci_low', 'ci_high', 'win', 'tie', 'loss', 'mean_steps_delta', 'method', 'note']
    for i, r in enumerate(comp_rows):
        r['csv_row'] = 'comparisons.csv:%d' % (i + 2)
    write_csv(os.path.join(OUTDIR, 'comparisons.csv'), comp_rows, comp_fields + ['csv_row'])

    # ---- equivalence / parity ----
    plain = per_arm_seed['parity-plain(200g,130steps)'][1017]
    comp = per_arm_seed['parity-compiled(200g,130steps)'][1017]
    par = drift('compiled - plain (200 graphs, fixed 130 steps)', comp, plain)
    # default-path regression: parity-plain vs t24 fixed-130 artifact on the common 200 graphs
    t24 = load_cell('outputs/v11/t7/curve_k200_s1017_step130', 1.0)
    common = sorted(set(plain) & set(t24))
    reg = drift('parity-plain - t24 curve130 (default path, common graphs)', plain, t24, common)
    eq_rows = [dict(graph_id=g, plain=plain[g]['search_size'], compiled=comp[g]['search_size'],
                    delta=comp[g]['search_size'] - plain[g]['search_size'],
                    steps_plain=plain[g]['steps'], steps_compiled=comp[g]['steps'])
               for g in sorted(set(plain) & set(comp))]
    for i, r in enumerate(eq_rows):
        r['csv_row'] = 'equivalence.csv:%d' % (i + 2)
    write_csv(os.path.join(OUTDIR, 'equivalence.csv'), eq_rows,
              ['graph_id', 'plain', 'compiled', 'delta', 'steps_plain', 'steps_compiled', 'csv_row'])
    print('parity compiled-plain: identical=%d/200 max|d|=%s ge2=%d ge3=%d'
          % (par['identical'], par['max_abs'], par['n_ge2'], par['n_ge3']))
    print('default-path regression vs t24: identical=%d/%d max|d|=%s'
          % (reg['identical'], reg['n_graphs'], reg['max_abs']))
    eq_summary = [
        dict(check='equiv_identical_graphs', value=par['identical'], unit='graphs',
             n_graphs=par['n_graphs'], ids='', note='compiled vs plain 逐图相同（200 图，固定 130 步）'),
        dict(check='equiv_max_abs_delta', value=par['max_abs'], unit='indep_nodes',
             n_graphs=par['n_graphs'], ids='', note='compiled vs plain 逐图最大 |Δ|'),
        dict(check='equiv_graphs_abs_ge_2', value=par['n_ge2'], unit='graphs',
             n_graphs=par['n_graphs'], ids=';'.join(par['ge2_ids']),
             note='compiled vs plain |Δ|≥2 图数（门槛 ≤1%）'),
        dict(check='equiv_graphs_abs_ge_3', value=par['n_ge3'], unit='graphs',
             n_graphs=par['n_graphs'], ids=';'.join(par['ge3_ids']),
             note='compiled vs plain |Δ|≥3 图数（任一即失败）'),
        dict(check='equiv_default_path_identical', value=reg['identical'], unit='graphs',
             n_graphs=reg['n_graphs'], ids='', note='默认路径回归：plain vs t24 curve130 相同图数'),
        dict(check='equiv_default_path_max_abs', value=reg['max_abs'], unit='indep_nodes',
             n_graphs=reg['n_graphs'], ids='', note='默认路径回归最大 |Δ|（默认关应恒 0）'),
    ]
    for i, r in enumerate(eq_summary):
        r['csv_row'] = 'equivalence_summary.csv:%d' % (i + 2)
    write_csv(os.path.join(OUTDIR, 'equivalence_summary.csv'), eq_summary,
              ['check', 'value', 'unit', 'n_graphs', 'ids', 'note', 'csv_row'])

    # ---- reconciliation vs t21 JSONs ----
    recon = []
    def cmp_json(relpath, path_getter, label):
        p = rel(relpath)
        if not os.path.exists(p):
            recon.append(dict(reference=relpath, status='MISSING'))
            return
        d = json.load(open(p, encoding='utf-8'))
        for arm, val in path_getter(d):
            mine = pooled.get(arm) or per_arm_seed.get(arm, {})
            if isinstance(mine, dict) and mine and isinstance(next(iter(mine.values())), dict):
                mv = sum(v['search_size'] for v in mine.values()) / len(mine)
            else:
                mv = None
            recon.append(dict(reference=relpath, arm=arm, theirs=val, mine=mv,
                              diff=None if mv is None else mv - val,
                              status='PASS' if mv is not None and abs(mv - val) < 5e-4 else
                                     ('SKIP' if mv is None else 'FAIL')))
    cmp_json('outputs/v11/r2/p05/ab_three_arm.json',
             lambda d: [('C(lead on,PAVE on)', d['arms']['C']['pooled']['search_size']['mean']),
                        ('A(lead off,PAVE on)', d['arms']['A']['pooled']['search_size']['mean']),
                        ('B(lead off,PAVE off)', d['arms']['B']['pooled']['search_size']['mean'])],
             'G1')
    cmp_json('outputs/v11/r2/p05/three_way.json',
             lambda d: [('B(lead off,PAVE off)', d['arms']['B']['search_size']['mean']),
                        ('B+(fused)', d['arms']['B+']['search_size']['mean']),
                        ('A0(1s,PAVE on)', d['A0_1s']['search'])],
             'G3')
    cmp_json('outputs/v11/r2/p05/ab01_two_arm.json',
             lambda d: [("A'(lead off,PAVE on)", d['arms']['A']['search_size']['mean']),
                        ("B'(lead off,PAVE off)", d['arms']['B']['search_size']['mean'])],
             'G2')
    cmp_json('outputs/v11/r2/p05/l3_three_seed.json',
             lambda d: [('L3(1s,fused)', d['pooled_search'])], 'L3')
    worst = max((abs(r['diff']) for r in recon if r.get('diff') is not None), default=0.0)
    fails = [r for r in recon if r['status'] == 'FAIL']
    print('reconciliation vs t21 JSON: %d checks, max|diff|=%.6f -> %s'
          % (len(recon), worst, 'FAIL' if fails else 'PASS'))
    for r in recon:
        print('   ', r.get('reference'), r.get('arm'), 'theirs=%s mine=%s diff=%s %s'
              % (r.get('theirs'), r.get('mine'), r.get('diff'), r['status']))

    # ---- dump ----
    write_csv(os.path.join(OUTDIR, 'v13_rows_merged.csv'), merged,
              ['arm', 'block', 'budget', 'seed', 'graph_id', 'search_size', 'final_size', 'steps',
               'ms_per_step', 'wall_ms', 'search_ms', 'pave_gain', 'dtype', 'dense', 'fp32_csr',
               'fixed_steps', 'skip_incumbent', 'no_pave', 'compile_model'])
    out = dict(
        generated_at='2026-11-18', task='t25',
        include_prefixes=INCLUDE_PREFIXES,
        exclude_prefixes=[dict(pattern=p, reason=w) for p, w in EXCLUDE_PREFIXES],
        bootstrap=dict(reps=REPS, seed=SEED, unit='graph (cluster), carries all seeds',
                       percentile=[2.5, 97.5]),
        arms={a['arm']: dict(block=a['block'], budget_s=a['budget'], report_only=a['report_only'],
                             note=a['note'], dirs={str(k): v for k, v in a['seed_dirs'].items()},
                             cells={'%s' % s: stats['%s|%s' % (a['arm'], s)] for s in a['seed_dirs']},
                             pooled=next((p for p in pooled_rows if p['arm'] == a['arm']), None))
              for a in ARMS},
        comparisons=comp_rows,
        equivalence=dict(
            compiled_vs_plain={k: par[k] for k in ('n_graphs', 'identical', 'within_one', 'n_ge2',
                                                   'n_ge3', 'max_abs', 'mean', 'ge2_ids', 'ge3_ids')},
            default_path_regression={k: reg[k] for k in ('n_graphs', 'identical', 'within_one',
                                                         'n_ge2', 'n_ge3', 'max_abs', 'mean',
                                                         'ge2_ids', 'ge3_ids')}),
        reconciliation=dict(checks=recon, max_abs_diff=worst, status='FAIL' if fails else 'PASS'),
    )
    with open(os.path.join(OUTDIR, 'ab_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('wrote', OUTDIR)
    if fails:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
