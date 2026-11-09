#!/usr/bin/env node
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const scratch = path.join(root, 'tmp/v5_reporting');
const analysis = path.join(root, 'outputs/v5_frequency/analysis');
const destination = path.join(root, 'docs/实验结果.xlsx');
const packages = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
await fs.mkdir(scratch, { recursive: true });
try {
  await fs.symlink(packages, path.join(scratch, 'node_modules'), process.platform === 'win32' ? 'junction' : 'dir');
} catch (error) {
  if (error.code !== 'EEXIST') throw error;
}
const require = createRequire(path.join(scratch, 'package.json'));
const { SpreadsheetFile, FileBlob } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const book = await SpreadsheetFile.importXlsx(await FileBlob.load(destination));
const plans = [['V5汇总', 'summary.csv'], ['V5逐种子', 'per_seed.csv'], ['V5逐图配对', 'paired_effects.csv']];
const names = new Set(plans.map(([name]) => name));
const preserved = book.worksheets.items.filter(sheet => !names.has(sheet.name)).map(sheet => ({
  name: sheet.name, values: JSON.stringify(sheet.getUsedRange()?.values),
  formulas: JSON.stringify(sheet.getUsedRange()?.formulas),
}));
const labels = {
  protocol: '协议', precision: '精度', post: '后处理', budget_s: '预算 (s)', method: '方法',
  seed: '种子', seed_count: '种子数', graph_count: '图数', graphs: '图数', mean_size: 'Mean Size',
  size_seed_sd: '种子 SD', mean_solve_s: '平均求解 (s)', p95_s: 'P95 (s)', feasible_rate: '可行率',
  params: 'Params', peak_memory_mib: '峰值显存 (MiB)', mean_steps: '实际搜索步数', mean_post_s: '后处理 (s)',
  deadline_overruns: '超时数', max_overrun_s: '最大超时 (s)', candidate: '候选', control: '对照',
  mean_delta: 'ΔSize', ci_low: '图级 95% CI 下界', ci_high: '图级 95% CI 上界', positive_seeds: '正向种子数',
  delta_baseline: 'Δ原骨干',
  seed_deltas: '分种子 ΔSize', wins: '胜', ties: '平', losses: '负',
};

function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (ch === ',' && !quoted) {
      row.push(field); field = '';
    } else if (ch === '\n' && !quoted) {
      row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = '';
    } else field += ch;
  }
  if (row.length || field) rows.push([...row, field]);
  return rows;
}

for (const [name, filename] of plans) {
  const [headers, ...rows] = parseCsv(await fs.readFile(path.join(analysis, filename), 'utf8'));
  if (!rows.length) throw new Error(`Missing evaluation rows in ${filename}`);
  const sheet = book.worksheets.items.find(item => item.name === name) ?? book.worksheets.add(name);
  sheet.getUsedRange()?.clear({ applyTo: 'contents' });
  const last = String.fromCharCode(64 + headers.length);
  const cells = sheet.getRange(`A1:${last}${rows.length + 1}`);
  cells.values = [headers.map(key => labels[key] ?? key), ...rows.map(row =>
    row.map(value => value === '' ? null : Number.isFinite(Number(value)) ? Number(value) : value))];
  cells.format.font.name = 'Arial';
  cells.format.font.size = 10;
  cells.format.rowHeight = 20;
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  const header = sheet.getRange(`A1:${last}1`);
  header.format.fill = '#243B53';
  header.format.font.color = '#FFFFFF';
  header.format.font.bold = true;
  header.format.rowHeight = 30;
  header.format.wrapText = true;
  headers.forEach((key, index) => {
    const col = String.fromCharCode(65 + index);
    sheet.getRange(`${col}1`).format.columnWidth = ['method', 'candidate', 'control', 'seed_deltas'].includes(key) ? 30 : 18;
    if (key === 'feasible_rate') sheet.getRange(`${col}2:${col}${rows.length + 1}`).setNumberFormat('0.0%');
    if (['mean_size', 'size_seed_sd', 'mean_delta', 'delta_baseline', 'ci_low', 'ci_high', 'mean_solve_s', 'p95_s', 'mean_post_s', 'max_overrun_s'].includes(key)) {
      sheet.getRange(`${col}2:${col}${rows.length + 1}`).setNumberFormat('0.0000');
    }
  });
}
book.recalculate();
for (const previous of preserved) {
  const cells = book.worksheets.getItem(previous.name).getUsedRange();
  if (JSON.stringify(cells?.values) !== previous.values || JSON.stringify(cells?.formulas) !== previous.formulas) {
    throw new Error(`Existing worksheet changed: ${previous.name}`);
  }
}
for (const name of names) {
  const preview = await book.render({ sheetName: name, range: 'A1:K12', scale: 1 });
  await fs.writeFile(path.join(scratch, `${name}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const staged = path.join(scratch, '实验结果.xlsx');
await (await SpreadsheetFile.exportXlsx(book)).save(staged);
await fs.rename(staged, destination);
console.log(JSON.stringify({ updated_sheets: [...names] }));
