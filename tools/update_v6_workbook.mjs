#!/usr/bin/env node
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const scratch = path.join(root, 'tmp/v6_reporting');
const workbook = path.join(root, 'docs/实验结果.xlsx');
const packages = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
await fs.mkdir(scratch, { recursive: true });
try {
  await fs.symlink(packages, path.join(scratch, 'node_modules'), process.platform === 'win32' ? 'junction' : 'dir');
} catch (error) {
  if (error.code !== 'EEXIST') throw error;
}
const require = createRequire(path.join(scratch, 'package.json'));
const { SpreadsheetFile, FileBlob } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const result = JSON.parse(await fs.readFile(path.join(root, 'outputs/v6/analysis/pilot.json'), 'utf8'));
const book = await SpreadsheetFile.importXlsx(await FileBlob.load(workbook));
const name = 'V6 SC-RLNN';
const sheet = book.worksheets.items.find(item => item.name === name) ?? book.worksheets.add(name);
sheet.getUsedRange()?.unmerge();
sheet.getUsedRange()?.clear({ applyTo: 'contents' });
sheet.showGridLines = false;

const method = ({ arm }) => arm === 'full' ? 'SC-RLNN' : 'Backbone B';
const post = ({ post }) => post === 'pave' ? 'A.3 + PAVE' : 'A.3';
const verdict = value => value ? '通过' : '未通过';
const quality = [
  ['Mean Size', 'mean_size', '0.0000'], ['跨种子 SD', 'size_seed_sd', '0.0000'],
  ['ΔSize vs B', 'delta_size', '0.0000'], ['平均求解 (s)', 'mean_solve_s', '0.0000'],
  ['可行率', 'feasible_rate', '0.0%'], ['Params', 'params', '#,##0'],
  ['峰值显存 (MiB)', 'peak_memory_mib', '0.0'],
];
let nextRow = 1;
function section(title, columns, rows) {
  const last = String.fromCharCode(64 + columns.length);
  const heading = sheet.getRange(`A${nextRow}:${last}${nextRow}`);
  heading.merge();
  heading.format.fill = '#243B53';
  heading.format.font.color = '#FFFFFF';
  heading.format.font.bold = true;
  heading.format.rowHeight = 24;
  sheet.getRange(`A${nextRow}`).values = [[title]];
  const start = nextRow + 1;
  const end = start + rows.length;
  const data = sheet.getRange(`A${start}:${last}${end}`);
  const values = [columns.map(([label]) => label), ...rows.map(row =>
    columns.map(([, getter]) => (typeof getter === 'function' ? getter(row) : row[getter]) ?? null))];
  data.values = values;
  data.format.rowHeight = 21;
  data.format.font.name = 'Arial';
  data.format.font.size = 10;
  const header = sheet.getRange(`A${start}:${last}${start}`);
  header.format.fill = '#E6EDF4';
  header.format.font.bold = true;
  header.format.wrapText = true;
  header.format.rowHeight = 32;
  columns.forEach(([, , format], index) => {
    const column = String.fromCharCode(65 + index);
    if (format) sheet.getRange(`${column}${start + 1}:${column}${end}`).setNumberFormat(format);
  });
  if (sheet.getRange(`A${start + 1}:${last}${end}`).values.flat().join('|') !== values.slice(1).flat().join('|')) {
    throw new Error(`Worksheet data mismatch in ${title}`);
  }
  nextRow = end + 2;
}

section('固定 100 步 · 100 张验证图', [
  ['方法', method], ['解码', post], ['训练种子数', 'seeds'], ['每种子图数', 'graphs'], ...quality,
], result.summary);
section('逐种子', [
  ['方法', method], ['解码', post], ['配对种子', 'seed'], ['图数', 'graphs'],
  quality[0], ['图间 SD', 'graph_sd', '0.0000'], ...quality.slice(2),
], result.per_seed);
section('图级配对 · SC-RLNN 相对 B', [
  ['解码', post], ['种子', 'seed'], ['图数', 'graphs'], ['ΔSize', 'delta_size', '0.0000'],
  ['胜', 'wins'], ['平', 'ties'], ['负', 'losses'],
], result.paired);
section('100 步独占测速', [
  ['执行模式', 'method'], ['精度', 'precision'], ['图数', 'graphs'], ['重复次数', 'repeats'],
  ['B (s)', 'b_s', '0.0000'], ['SC-RLNN (s)', 'full_s', '0.0000'],
  ['耗时比', 'ratio', '0.00"x"'], ['≤1.08×', row => verdict(row.speed_pass)],
], result.profiles);
section('分支与逐层诊断', [
  ['方法', method], ['种子', 'seed'], ['图数', 'graphs'],
  ['中带 logit 能量', 'meso_logit_energy', '0.0000'], ['高带 logit 能量', 'high_logit_energy', '0.0000'],
  ...[1, 2, 3, 4, 5].map(layer => [`第 ${layer} 层能量比`, `layer${layer}_dirichlet_ratio`, '0.0000']),
], result.health);
section('首轮训练', [
  ['种子', 'seed'], ['Epoch', 'completed_epochs'], ['更新数', 'updates'],
  ['最佳验证均值', 'best_validation_size', '0.0000'],
  ['峰值显存 (MiB)', 'peak_memory_mib', '0.0'], ['训练总耗时 (s)', 'train_wall_s', '0.0'],
], result.train);
const gateLabels = { speed: '速度', bare_size: '裸解码质量', pave_all_seeds: '三种子 PAVE',
  high_energy: '高带 logit 能量', deep_dirichlet: '第五层狄利克雷能量比' };
section('准入判断', [['门槛', 'gate'], ['结果', row => verdict(row.pass)]],
  Object.entries(result.gates).map(([key, pass]) => ({ gate: gateLabels[key], pass })));

sheet.getRange(`A1:K${nextRow}`).format.columnWidth = 17;
sheet.getRange(`A1:A${nextRow}`).format.columnWidth = 24;
sheet.freezePanes.freezeRows(2);
book.recalculate();
const preview = await book.render({ sheetName: name, range: 'A1:K22', scale: 1 });
await fs.writeFile(path.join(scratch, 'v6.png'), new Uint8Array(await preview.arrayBuffer()));
const staged = path.join(scratch, '实验结果.xlsx');
await (await SpreadsheetFile.exportXlsx(book)).save(staged);
await fs.copyFile(staged, workbook);
console.log(name);
