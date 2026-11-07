#!/usr/bin/env node
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/package.json'));
const { SpreadsheetFile, FileBlob } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const output = path.join(root, 'outputs', 'v1');
const destination = path.join(root, 'docs', '实验结果.xlsx');
const book = await SpreadsheetFile.importXlsx(await FileBlob.load(destination));
const average = values => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;

const runs = [];
const training = [];
const diagnostics = [];
for (const entry of (await fs.readdir(output, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
  if (!entry.isDirectory() || entry.name.startsWith('smoke_')) continue;
  const relative = 'outputs/v1/' + entry.name;
  const filename = path.join(root, relative, 'run.json');
  let meta;
  try {
    meta = JSON.parse(await fs.readFile(filename, 'utf8'));
  } catch (error) {
    if (error.code === 'ENOENT') continue;
    throw error;
  }
  if (meta.status !== 'completed') continue;
  if (meta.mode === 'train' || meta.mode === 'value_train') {
    const config = meta.config;
    training.push([meta.run, meta.method ?? config.frequency ?? 'RLNN', config.seed, config.epochs,
      config.graph_batch_size ?? config.batch_size, config.graph_batch_size ? config.batch_size : config.mini_batch_size ?? null,
      meta.wall_time_s / 60, meta.generation_time_s == null ? null : meta.generation_time_s / 60,
      meta.val_mae ?? null, meta.val_zero_mae ?? null, meta.val_delta ?? null, meta.val_positive_rate ?? null, relative]);
    continue;
  }
  if (meta.mode !== 'eval' || (meta.samples !== 1000 && !meta.config.diagnostic)) continue;
  const csv = (await fs.readFile(path.join(root, relative, 'per_graph.csv'), 'utf8')).trimEnd().split(/\r?\n/);
  const columns = csv.shift().split(',');
  const id = columns.indexOf('graph_id');
  const size = columns.indexOf('independent_set_size');
  const extra = columns.indexOf('extra_candidates');
  const gain = columns.indexOf('archive_gain');
  const archiveSeconds = columns.indexOf('archive_seconds');
  const records = csv.map(row => row.split(','));
  if (meta.config.diagnostic) {
    const rawGap = columns.indexOf('baseline_archive_gap');
    const qualityGap = columns.indexOf('archive_gap');
    diagnostics.push([meta.run, meta.samples, average(records.map(row => Number(row[rawGap]))),
      average(records.map(row => Number(row[qualityGap]))), average(records.map(row => Number(row[gain]))),
      average(records.map(row => Number(row[extra]))), relative]);
    continue;
  }
  const perGraph = new Map(records.map(row => [row[id], Number(row[size])]));
  if (perGraph.size !== 1000) throw new Error(relative + ' must contain 1,000 unique graph IDs');
  runs.push({ ...meta, relative, perGraph, extra: average(records.map(row => Number(row[extra] || 0))),
    archiveSeconds: average(records.map(row => Number(row[archiveSeconds] || 0))),
    archiveGain: gain >= 0 ? average(records.filter(row => row[gain] !== '').map(row => Number(row[gain]))) : null,
    budget: meta.config.time_budget_s ?? 0 });
}

const baselines = new Map(runs.filter(run => run.method === 'RLNN').map(run => [
  [run.train_seed, run.search_seed, run.budget].join('/'), run.perGraph,
]));
for (const run of runs) {
  const baseline = baselines.get([run.train_seed, run.search_seed, run.budget].join('/')) ??
    (run.method === 'Greedy' ? baselines.get([run.train_seed, 1017, run.budget].join('/')) : null);
  run.delta = baseline && [...run.perGraph.keys()].every(id => baseline.has(id))
    ? average([...run.perGraph].map(([id, size]) => size - baseline.get(id))) : null;
}

function sheet(name, headers, data, widths, formats = {}) {
  const tab = book.worksheets.items.find(item => item.name === name) ?? book.worksheets.add(name);
  tab.getUsedRange()?.clear({ applyTo: 'contents' });
  const last = String.fromCharCode(64 + headers.length);
  const end = Math.max(2, data.length + 1);
  tab.getRange('A1:' + last + (data.length + 1)).values = [headers, ...data];
  tab.showGridLines = false;
  tab.freezePanes.freezeRows(1);
  const range = tab.getRange('A1:' + last + end);
  range.format.font.name = 'Arial';
  range.format.font.size = 10;
  range.format.rowHeight = 20;
  const header = tab.getRange('A1:' + last + '1');
  header.format.fill = '#243B53';
  header.format.font.color = '#FFFFFF';
  header.format.font.bold = true;
  header.format.rowHeight = 26;
  for (const [i, width] of widths.entries()) {
    const letter = String.fromCharCode(65 + i);
    tab.getRange(letter + ':' + letter).format.columnWidth = width;
    if (formats[letter]) tab.getRange(letter + '2:' + letter + end).setNumberFormat(formats[letter]);
  }
}

const groups = new Map();
for (const run of runs) {
  const key = run.method + '|' + run.budget;
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push(run);
}
const summary = [...groups.values()].map(group => {
  const first = group[0];
  const n = group.length;
  const mean = average(group.map(run => run.mean_size));
  const std = n > 1 ? Math.sqrt(group.reduce((s, run) => s + (run.mean_size - mean) ** 2, 0) / (n - 1)) : null;
  return [first.method, std === null ? mean.toFixed(3) : mean.toFixed(3) + ' ± ' + std.toFixed(3),
    group.every(run => run.delta !== null) ? average(group.map(run => run.delta)) : null,
    average(group.map(run => run.mean_solve_time_s)), average(group.map(run => run.feasible_rate)),
    first.params, Math.max(...group.map(run => run.peak_memory_mib)), n, first.budget || null, mean, std];
}).sort((a, b) => (a[8] ?? 0) - (b[8] ?? 0) || (a[0] === 'RLNN' ? -1 : b[0] === 'RLNN' ? 1 : a[0].localeCompare(b[0])));
sheet('V1汇总', ['方法', 'Mean Size ↑（均值±标准差）', 'ΔSize ↑', '平均求解时间 ↓ (s)', '可行率 ↑',
  'Params', '峰值显存 (MiB)', '种子数', '时限 (s)', '均值', '样本标准差'],
  summary, [31, 30, 12, 22, 12, 13, 17, 11, 11, 13, 14],
  { C: '+0.000;-0.000;0.000', D: '0.0000', E: '0.00%', F: '#,##0', G: '0.00', J: '0.000', K: '0.000' });
sheet('V1逐种子', ['方法', '训练种子', '搜索种子', 'Mean Size', '配对 ΔSize', '单图秒', '可行率',
  'Params', '峰值 MiB', '时限 (s)', '附加候选/图', '额外解码秒/图', '保存增益/图', '设备', 'GPU', '数据来源'],
  runs.map(run => [run.method, run.train_seed, run.search_seed, run.mean_size, run.delta,
    run.mean_solve_time_s, run.feasible_rate, run.params, run.peak_memory_mib, run.budget || null,
    run.extra, run.archiveSeconds, run.archiveGain, run.device, run.gpu ?? null, run.relative]),
  [31, 12, 12, 14, 14, 12, 11, 13, 13, 12, 15, 18, 15, 29, 9, 47],
  { D: '0.000', E: '0.000', F: '0.0000', G: '0.00%', H: '#,##0', I: '0.00', K: '0.0', L: '0.0000', M: '0.000' });
sheet('V1训练', ['任务', '模型或分频', '训练种子', 'Epoch', '图批', '状态批', '训练分钟', '配对样本生成分钟',
  '验证 MAE', '零预测 MAE', '平均配对收益', '正收益率', '数据来源'],
  training, [32, 17, 13, 11, 10, 10, 14, 23, 15, 15, 17, 12, 46],
  { G: '0.00', H: '0.00', I: '0.0000', J: '0.0000', K: '0.0000', L: '0.00%' });
if (diagnostics.length) {
  sheet('V1候选诊断', ['任务', '图数', '原档案平均差距', '增广档案平均差距', '平均保存增益', '额外候选/图', '数据来源'],
    diagnostics, [28, 11, 20, 22, 18, 18, 43], { C: '0.000', D: '0.000', E: '0.000' });
}

const paper = [
  ['Gurobi', 'OR', 19.98, 47.57 * 60], ['KaMIS', 'OR', 20.10, 1.40 * 3600],
  ['PPO/LwD', 'RL', 19.01, 1.28 * 60], ['INTEL', 'SL', 18.47, 13.07 * 60],
  ['DGL', 'SL', 17.36, 12.78 * 60], ['DIMES', 'RL', null, null],
  ['DIFUSCO', 'SL', 18.52, 16.05 * 60], ['LTFT', 'UL', 19.18, 32],
  ['DiffUCO', 'UL', 19.24, 54], ['iSCO', 'H', 19.29, 2.71 * 60],
  ['RLNN', 'UL', 19.52, 1.64 * 60], ['RLSA', 'H', 19.97, 35],
];
sheet('RLNN论文表1', ['方法', '类别', 'RB200–300 Size', '秒/500图', '来源'],
  paper.map(row => [...row, 'RLNN（ICML 2025）表 1']), [21, 12, 22, 18, 32], { C: '0.00', D: '0.00' });

book.recalculate();
const inspection = await book.inspect({ kind: 'table', range: 'V1汇总!A1:K8', tableMaxRows: 8, tableMaxCols: 11 });
console.log(inspection.ndjson);
const preview = await book.render({ sheetName: 'V1汇总', range: 'A1:K' + Math.min(summary.length + 2, 16), scale: 1 });
await fs.writeFile(path.join(root, 'tmp', 'v1_results.png'), new Uint8Array(await preview.arrayBuffer()));
const result = await SpreadsheetFile.exportXlsx(book);
const staged = path.join(root, 'tmp', 'v1_results.xlsx');
await result.save(staged);
await fs.rename(staged, destination);
console.log('V1汇总：' + summary.length + ' 组，' + runs.length + ' 次完整评估');
