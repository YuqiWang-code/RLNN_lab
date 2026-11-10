#!/usr/bin/env node
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const scratch = path.join(root, 'tmp/mamba_reporting');
const result = path.join(root, 'outputs/v7_context');
const workbook = path.join(root, 'docs/实验结果.xlsx');
await fs.mkdir(scratch, { recursive: true });
const packages = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
try {
  await fs.symlink(packages, path.join(scratch, 'node_modules'), process.platform === 'win32' ? 'junction' : 'dir');
} catch (error) {
  if (error.code !== 'EEXIST') throw error;
}
const require = createRequire(path.join(scratch, 'package.json'));
const { SpreadsheetFile, FileBlob } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const read = async name => JSON.parse(await fs.readFile(path.join(result, name), 'utf8'));
const decision = await read('analysis/p0.json');
const names = await fs.readdir(path.join(result, 'preflight'));
const profiles = await Promise.all(names.filter(name => /^profile_\d+_(no|fp16)/.test(name)).map(async name => {
  const data = await read(`preflight/${name}/run.json`);
  const b = data.timings.baseline;
  const m = data.timings.mamba;
  const version = name.endsWith('_initial') ? '原版' : name.endsWith('_repeat') ? '复测' : '优化';
  return [version, data.width, data.precision === 'no' ? 'FP32' : 'FP16', data.samples, data.repeats,
    b.bare_s*1000, m.bare_s*1000, data.ratio, m.static_features_s*1000, m.context_s*1000, m.rollout_s*1000];
}));
const budget = await read(`preflight/budget_${decision.width}_optimized_no/run.json`);
const probe = await read('preflight/information_probe/run.json');
const smoke = await Promise.all(['mamba', 'message'].map(async arm => {
  const d = await read(`preflight/train_smoke_${arm}/run.json`);
  return [arm, d.updates, d.params, d.trainable_params, d.peak_memory_mib, d.train_wall_s,
    d.first_gradients[0].alpha_grad, d.first_gradients[1].encoder_grad_norm];
}));
const book = await SpreadsheetFile.importXlsx(await FileBlob.load(workbook));
const sheetName = 'Mamba v0 P0';
const sheet = book.worksheets.items.find(s => s.name === sheetName) ?? book.worksheets.add(sheetName);
sheet.getUsedRange()?.unmerge();
sheet.getUsedRange()?.clear({ applyTo: 'contents' });
sheet.showGridLines = false;
let row = 1;
function table(title, headers, values, formats = {}) {
  const last = String.fromCharCode(64 + headers.length);
  const titleRange = sheet.getRange(`A${row}:${last}${row}`);
  titleRange.merge();
  titleRange.format.fill = '#243B53';
  titleRange.format.font.color = '#FFFFFF';
  titleRange.format.font.bold = true;
  titleRange.format.rowHeight = 24;
  sheet.getRange(`A${row++}`).values = [[title]];
  const head = sheet.getRange(`A${row}:${last}${row}`);
  head.values = [headers];
  head.format.fill = '#E6EDF4';
  head.format.font.bold = true;
  head.format.wrapText = true;
  head.format.rowHeight = 31;
  const first = row + 1;
  row += values.length;
  sheet.getRange(`A${first}:${last}${row}`).values = values;
  for (const [col, format] of Object.entries(formats)) {
    sheet.getRange(`${col}${first}:${col}${row}`).setNumberFormat(format);
  }
  row += 2;
}
table('Mamba v0 · 训练前准入', ['状态', '宽度', '正确性', 'FP32 耗时比', 'FP16 耗时比', '代理重复性 Spearman'],
  [[decision.status, decision.width, decision.correctness ? '通过' : '未通过',
    decision.speed_ratios.no, decision.speed_ratios.fp16, decision.proxy_reliability]], { D: '0.000', E: '0.000', F: '0.0000' });
table('20 张训练图 · 100 步 + A.3 · 零残差 · 单卡独占', ['版本', '宽度', '精度', '图数', '重复',
  'B (ms)', 'Mamba (ms)', '耗时比', '静态特征 (ms)', '编码 (ms)', '回放 (ms)'], profiles,
  { F: '0.00', G: '0.00', H: '0.000', I: '0.00', J: '0.00', K: '0.00' });
table('同预算步数 · 零残差 · 20 张训练图', ['预算 (s)', '方法', '平均步数', '平均耗时 (s)', '超时率', '回退率'],
  Object.entries(budget.results).flatMap(([seconds, runs]) => Object.entries(runs).map(([arm, data]) =>
    [Number(seconds), arm, data.search_steps, data.solve_s, data.overrun, data.fallback])),
  { A: '0.0', C: '0.0', D: '0.000', E: '0.0%', F: '0.0%' });
table('跨图代理 · 20 张训练来源留出图', ['方法', 'AUC', 'Spearman', 'Brier', 'Mamba ΔAUC', '95% CI 下界', '95% CI 上界'],
  Object.entries(probe.mean_metrics).map(([arm, data]) => {
    const d = probe.paired_delta_auc[arm];
    return [arm, data.auc, data.spearman, data.brier, d?.delta_auc ?? null, d?.ci95[0] ?? null, d?.ci95[1] ?? null];
  }), { B: '0.0000', C: '0.0000', D: '0.0000', E: '0.0000', F: '0.0000', G: '0.0000' });
table('训练入口冒烟 · 8 图 · 2 次更新', ['方法', '更新数', 'Params', '可训练 Params', '峰值显存 (MiB)',
  '耗时 (s)', '首步门幅梯度', '第二步编码梯度'], smoke,
  { C: '#,##0', D: '#,##0', E: '0.0', F: '0.00', G: '0.0000', H: '0.000000' });
sheet.getRange(`A1:K${row}`).format.columnWidth = 17;
sheet.getRange(`A1:A${row}`).format.columnWidth = 25;
sheet.freezePanes.freezeRows(2);
book.recalculate();
const preview = await book.render({ sheetName, range: `A1:K${row - 1}`, scale: 1 });
await fs.writeFile(path.join(scratch, 'p0.png'), new Uint8Array(await preview.arrayBuffer()));
const staged = path.join(scratch, '实验结果.xlsx');
await (await SpreadsheetFile.exportXlsx(book)).save(staged);
await fs.copyFile(staged, workbook);
console.log(sheetName);
