#!/usr/bin/env node
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/package.json'));
const { SpreadsheetFile, FileBlob } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const destination = path.join(root, 'docs', '实验结果.xlsx');
const book = await SpreadsheetFile.importXlsx(await FileBlob.load(destination));
const average = numbers => numbers.length ? numbers.reduce((sum, value) => sum + value, 0) / numbers.length : null;
const postName = { none: '', rule: ' + 规则1→2', heuristic: ' + 启发式1→2', random: ' + 随机1→2',
  rule23: ' + 规则1→2/2→3', pave: ' + PAVE' };
const runs = [];
const training = [];
const diagnostics = [];

async function readCsv(file) {
  const lines = (await fs.readFile(file, 'utf8')).trimEnd().split(/\r?\n/);
  const columns = lines.shift().split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((value, i) => [columns[i], value])));
}

for (const entry of await fs.readdir(path.join(root, 'outputs', 'v2'), { recursive: true, withFileTypes: true })) {
  if (!entry.isFile() || entry.name !== 'run.json') continue;
  const folder = entry.parentPath;
  const relative = path.relative(root, folder).split(path.sep).join('/');
  if (relative.includes('/smoke_')) continue;
  let meta;
  try {
    meta = JSON.parse(await fs.readFile(path.join(folder, entry.name), 'utf8'));
  } catch (error) {
    if (error instanceof SyntaxError) continue;
    throw error;
  }
  if (meta.status !== 'completed') continue;
  if (meta.mode === 'eval') {
    const rows = await readCsv(path.join(folder, 'per_graph.csv'));
    const sizes = new Map(rows.map(row => [row.graph_id, Number(row.independent_set_size)]));
    if (sizes.size !== meta.samples) throw new Error(relative + ': incorrect graph count');
    runs.push({ ...meta, relative, sizes, name: meta.method + postName[meta.post],
      budget: meta.config.time_budget_s ?? 0,
      exchangeTime: average(rows.map(row => Number(row.exchange_time_s))),
      exchanges: average(rows.map(row => Number(row.exchanges))) });
  } else if (meta.mode === 'train') {
    const config = meta.config;
    const rows = (await readCsv(path.join(folder, 'metrics', 'metrics.csv'))).filter(row => row.epoch);
    const best = rows.reduce((a, b) => Number(b['val/independent_set_size']) > Number(a['val/independent_set_size']) ? b : a);
    training.push([meta.run, config.frequency, config.seed, config.warm_start ? '微调' : '从头',
      config.epochs, Number(rows.at(-1).step), meta.wall_time_s / 60, Number(best.epoch),
      Number(best['val/independent_set_size']), Number(best['train/energy']),
      Number(best['train/distance']), Number(best['train/flips']),
      Math.max(...rows.map(row => Number(row['train/peak_mib'] ?? 0))), relative]);
  } else if (meta.mode === 'opportunities') {
    for (const neighborhood of ['one_to_two', 'fallback']) {
      for (const [key, value] of Object.entries(meta[neighborhood])) {
        diagnostics.push(['交换机会', relative, neighborhood, key, value]);
      }
    }
  } else if (meta.mode === 'frequency_diagnostic') {
    for (const phase of meta.phases) {
      for (const [key, value] of Object.entries(phase)) {
        if (key !== 'step') diagnostics.push(['频域', relative, phase.step, key, value]);
      }
    }
    diagnostics.push(['频域', relative, null, 'paired_probes', meta.paired_probes],
      ['频域', relative, null, 'paired_spearman', meta.paired_spearman]);
  } else if (meta.mode === 'kamis_validation') {
    diagnostics.push(['KaMIS', relative, null, 'mean_size', meta.mean_size],
      ['KaMIS', relative, null, 'delta_vs_rlnn', meta.mean_delta],
      ['KaMIS', relative, null, 'mean_solve_time_s', meta.mean_solve_time_s]);
  } else if (meta.mode === 'value_train') {
    for (const [key, value] of Object.entries(meta.best)) {
      diagnostics.push(['PAVE', relative, null, key, value]);
    }
  }
}

function sheet(name, header, rows, widths, formats = {}) {
  if (!rows.length) return;
  const tab = book.worksheets.items.find(item => item.name === name) ?? book.worksheets.add(name);
  tab.getUsedRange()?.clear({ applyTo: 'contents' });
  const end = rows.length + 1;
  const last = String.fromCharCode(64 + header.length);
  const range = tab.getRange('A1:' + last + end);
  range.values = [header, ...rows];
  range.format.font.name = 'Arial';
  range.format.font.size = 10;
  range.format.rowHeight = 20;
  tab.showGridLines = false;
  tab.freezePanes.freezeRows(1);
  const firstRow = tab.getRange('A1:' + last + '1');
  firstRow.format.fill = '#243B53';
  firstRow.format.font.color = '#FFFFFF';
  firstRow.format.font.bold = true;
  firstRow.format.rowHeight = 26;
  for (const [index, width] of widths.entries()) {
    const column = String.fromCharCode(65 + index);
    tab.getRange(column + ':' + column).format.columnWidth = width;
    if (formats[column]) tab.getRange(column + '2:' + column + end).setNumberFormat(formats[column]);
  }
}

const key = run => [run.split, run.train_seed, run.search_seed, run.budget].join('/');
const baseline = new Map();
const shared = new Map();
for (const run of runs) {
  if (run.post === 'none') shared.set(path.posix.dirname(run.relative), run);
  if (run.name === 'RLNN' && (!baseline.has(key(run)) || !run.relative.includes('/diagnostics_'))) {
    baseline.set(key(run), run);
  }
}
const cohorts = new Map();
for (const run of runs.sort((a, b) => a.budget - b.budget || a.name.localeCompare(b.name) || a.train_seed - b.train_seed)) {
  const reference = run.relative.includes('/diagnostics_exchange')
    ? shared.get(path.posix.dirname(run.relative)) : baseline.get(key(run));
  if (reference && [...run.sizes.keys()].every(id => reference.sizes.has(id))) {
    const differences = [...run.sizes].map(([id, size]) => size - reference.sizes.get(id));
    run.delta = average(differences);
    run.wins = differences.filter(delta => delta > 0).length;
    run.ties = differences.filter(delta => delta === 0).length;
    run.losses = differences.filter(delta => delta < 0).length;
  }
  if (run.split !== 'test' || run.samples !== 1000) continue;
  const group = run.name + '/' + run.budget;
  if (!cohorts.has(group)) cohorts.set(group, []);
  cohorts.get(group).push(run);
}
const sum = numbers => numbers.reduce((total, value) => total + value, 0);
const scores = [...cohorts.values()].map(group => {
  const [first] = group;
  const n = group.length;
  const size = average(group.map(run => run.mean_size));
  const deviation = n > 1 ? Math.sqrt(sum(group.map(run => (run.mean_size - size) ** 2)) / (n - 1)) : null;
  const paired = group.every(run => run.delta !== undefined);
  return [first.name, deviation === null ? size.toFixed(3) : size.toFixed(3) + ' ± ' + deviation.toFixed(3),
    paired ? average(group.map(run => run.delta)) : null, average(group.map(run => run.mean_solve_time_s)),
    average(group.map(run => run.feasible_rate)), first.params, Math.max(...group.map(run => run.peak_memory_mib)),
    first.budget || null, n, ...['wins', 'ties', 'losses'].map(outcome => paired ? sum(group.map(run => run[outcome])) : null),
    average(group.map(run => run.group_times_s[0])), average(group.map(run => run.group_times_s[1])),
    average(group.map(run => run.exchange_gain)), average(group.map(run => run.exchangeTime)),
    first.post === 'none' ? null : sum(group.map(run => run.exchange_gain)) / (1000 * sum(group.map(run => run.exchangeTime))),
    sum(group.map(run => run.deadline_overruns))];
});
sheet('V2汇总', ['方法', 'Mean Size ↑（均值±标准差）', 'ΔSize ↑', '平均求解 (s) ↓', '可行率 ↑', 'Params',
  '峰值显存 (MiB)', '预算 (s)', '种子数', '胜', '平', '负', '前500总秒', '后500总秒', '交换增益/图', '交换秒/图', '增益/额外毫秒', '超时图数'],
  scores, [32, 26, 12, 17, 12, 13, 17, 10, 10, 10, 10, 10, 14, 14, 15, 15, 17, 13],
  { C: '+0.000;-0.000;0.000', D: '0.0000', E: '0.00%', F: '#,##0', G: '0.00', M: '0.00', N: '0.00',
    O: '0.000', P: '0.000000', Q: '0.0000' });
sheet('V2逐种子', ['方法', '划分', '训练种子', '搜索种子', '图数', '预算 (s)', 'Mean Size', 'ΔSize',
  '平均求解 (s)', '可行率', 'Params', '峰值 (MiB)', '胜', '平', '负', '前500总秒', '后500总秒', '交换增益/图',
  '交换秒/图', '交换次数/图', '超时图数', '来源'],
  runs.map(run => [run.name, run.split, run.train_seed, run.search_seed, run.samples, run.budget || null,
    run.mean_size, run.delta ?? null, run.mean_solve_time_s, run.feasible_rate, run.params, run.peak_memory_mib,
    run.wins ?? null, run.ties ?? null, run.losses ?? null, run.group_times_s[0], run.group_times_s[1] ?? null,
    run.exchange_gain, run.exchangeTime, run.exchanges, run.deadline_overruns, run.relative]),
  [32, 14, 11, 11, 10, 10, 14, 12, 16, 11, 13, 13, 9, 9, 9, 14, 14, 15, 15, 15, 13, 53],
  { G: '0.000', H: '+0.000;-0.000;0.000', I: '0.0000', J: '0.00%', K: '#,##0', L: '0.00', P: '0.00', Q: '0.00',
    R: '0.000', S: '0.000000', T: '0.000' });
sheet('V2训练', ['任务', '适配器', '种子', '初始化', 'Epoch', '更新数', '分钟', '最佳轮', '验证 Size',
  '期望能量', '距离正则', '期望翻转数', '峰值 (MiB)', '来源'],
  training.sort((a, b) => a[0].localeCompare(b[0])), [26, 20, 10, 12, 10, 11, 12, 12, 14, 14, 14, 16, 16, 48],
  { G: '0.00', I: '0.000', J: '0.0000', K: '0.0000', L: '0.000', M: '0.0' });
sheet('V2机制', ['诊断', '来源', '邻域或阶段', '指标', '数值'],
  diagnostics, [15, 43, 22, 25, 16], { E: '0.0000' });

const tab = scores.length ? 'V2汇总' : 'V2机制';
book.recalculate();
const inspection = await book.inspect({ kind: 'table', range: tab + '!A1:E6', include: 'values,formulas',
  tableMaxRows: 6, tableMaxCols: 5 });
console.log(inspection.ndjson);
const preview = await book.render({ sheetName: tab, range: scores.length ? 'A1:I12' : 'A1:E12', scale: 1 });
await fs.mkdir(path.join(root, 'tmp'), { recursive: true });
await fs.writeFile(path.join(root, 'tmp', 'v2_workbook_preview.png'), new Uint8Array(await preview.arrayBuffer()));
const updated = await SpreadsheetFile.exportXlsx(book);
const staged = path.join(root, 'tmp', 'v2_workbook.xlsx');
await updated.save(staged);
await fs.rename(staged, destination);
console.log('V2 test cohorts: ' + scores.length + ', scored runs: ' + runs.length + ', training runs: ' + training.length);
