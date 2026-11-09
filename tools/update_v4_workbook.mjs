#!/usr/bin/env node
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const scratch = path.join(root, 'tmp', 'v4_reporting');
const packages = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
await fs.mkdir(scratch, { recursive: true });
try {
  await fs.symlink(packages, path.join(scratch, 'node_modules'), process.platform === 'win32' ? 'junction' : 'dir');
} catch (error) {
  if (error.code !== 'EEXIST') throw error;
}
const require = createRequire(path.join(scratch, 'package.json'));
const { SpreadsheetFile, FileBlob } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const report = JSON.parse(await fs.readFile(path.join(root, 'outputs/v4_frequency/summary.json'), 'utf8'));
const destination = path.join(root, 'docs', '实验结果.xlsx');
const book = await SpreadsheetFile.importXlsx(await FileBlob.load(destination));
const names = ['V4汇总', 'V4逐种子', 'V4机制'];
const preserved = book.worksheets.items.filter(sheet => !names.includes(sheet.name)).map(sheet => ({
  name: sheet.name, values: JSON.stringify(sheet.getUsedRange()?.values),
  formulas: JSON.stringify(sheet.getUsedRange()?.formulas),
}));

function column(index) {
  let label = '';
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) label = String.fromCharCode(65 + (n - 1) % 26) + label;
  return label;
}

function writeSheet(name, headers, rows, widths, formats = {}) {
  const sheet = book.worksheets.items.find(item => item.name === name) ?? book.worksheets.add(name);
  sheet.getUsedRange()?.clear({ applyTo: 'contents' });
  const bottom = rows.length + 1;
  const last = column(headers.length - 1);
  const range = sheet.getRange(`A1:${last}${bottom}`);
  range.values = [headers, ...rows.map(row => row.map(value => value ?? null))];
  range.format.font.name = 'Arial';
  range.format.font.size = 10;
  range.format.rowHeight = 20;
  range.format.verticalAlignment = 'center';
  range.format.wrapText = false;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const header = sheet.getRange(`A1:${last}1`);
  header.format.fill = '#243B53';
  header.format.font.color = '#FFFFFF';
  header.format.font.bold = true;
  header.format.rowHeight = 28;
  header.format.horizontalAlignment = 'center';
  for (let i = 0; i < headers.length; i++) {
    const letter = column(i);
    sheet.getRange(`${letter}1:${letter}${bottom}`).format.columnWidth = widths[i] ?? 16;
    if (formats[letter] && rows.length) sheet.getRange(`${letter}2:${letter}${bottom}`).setNumberFormat(formats[letter]);
  }
  return sheet;
}

writeSheet('V4逐种子', ['方法（划分 / 精度 / 协议）', '划分', '精度', '协议', '骨干种子', '评分种子', '搜索种子',
  '图数', 'Mean Size', 'Δ原骨干', '平均求解 (s)', '可行率', 'Params', 'PyTorch峰值已分配显存 (MiB)',
  '初始Size', '规则后Size', '同运行后处理增益', '同运行平台增益', '平均搜索步数', '后处理秒/图', '评分秒/图',
  '预算 (s)', '持续搜索', '胜', '平', '负', '超时图数', '最大超时 (s)', '来源', '骨干权重', '评分权重', 'Δ参考来源'],
report.runs.map(run => [run.method, run.split, run.precision, run.protocol, run.backbone_seed, run.head_seed,
  run.search_seed, run.samples, run.mean_size, run.delta_base, run.mean_solve_time_s, run.feasible_rate,
  run.params, run.peak_memory_mib, run.mean_initial_size, run.mean_rule_size, run.base_gain, run.rule_gain,
  run.mean_search_steps, run.mean_post_time_s, run.mean_score_time_s, run.time_budget_s,
  run.extended_search ? '是' : '否', run.wins, run.ties, run.losses, run.deadline_overruns,
  run.max_deadline_overrun_s, run.source, run.backbone_checkpoint, run.value_checkpoint, run.baseline_source]),
[58, 13, 11, 18, 11, 11, 11, 10, 14, 15, 18, 12, 14, 30, 14, 14, 22, 22, 17, 17, 17, 11, 11, 10, 10, 10, 13, 17, 65, 65, 65, 65],
{ I: '0.00000', J: '+0.00000;-0.00000;0.00000', K: '0.00000', L: '0.00%', M: '#,##0', N: '0.00',
  O: '0.00000', P: '0.00000', Q: '0.00000', R: '0.00000', S: '0.00000', T: '0.00000', U: '0.00000',
  V: '0.00', AB: '0.00000' });

const summary = writeSheet('V4汇总', ['方法（划分 / 精度 / 协议）', 'Mean Size ↑（均值±标准差）', 'Δ原骨干 ↑',
  '平均求解 (s) ↓', '可行率 ↑', 'Params', 'PyTorch峰值已分配显存 (MiB)', '已完成种子数', '图数/种子',
  '同运行后处理增益', '同运行平台增益', '平均搜索步数', '胜', '平', '负', '超时图数', '最大超时 (s)',
  '划分', '精度', '协议', '预算 (s)', '持续搜索', '种子', '骨干来源', '评分来源'],
report.cohorts.map(cohort => [cohort.method, null, cohort.delta_base, cohort.mean_solve_time_s,
  cohort.feasible_rate, cohort.params, cohort.peak_memory_mib, cohort.seeds, cohort.samples,
  cohort.base_gain, cohort.rule_gain, cohort.mean_search_steps, cohort.wins, cohort.ties, cohort.losses,
  cohort.deadline_overruns, cohort.max_deadline_overrun_s, cohort.split, cohort.precision, cohort.protocol,
  cohort.time_budget_s, cohort.extended_search ? '是' : '否', cohort.seed_ids.join(', '),
  cohort.checkpoint_family, cohort.value_checkpoint_family]),
[58, 30, 15, 19, 13, 14, 30, 15, 13, 22, 22, 17, 10, 10, 10, 13, 17, 13, 11, 18, 11, 11, 13, 65, 65],
{ C: '+0.00000;-0.00000;0.00000', D: '0.00000', E: '0.00%', F: '#,##0', G: '0.00',
  J: '0.00000', K: '0.00000', L: '0.00000', Q: '0.00000', U: '0.00' });

for (const [index, cohort] of report.cohorts.entries()) {
  const row = index + 2;
  const references = letter => cohort.run_indices.map(runIndex => `'V4逐种子'!${letter}${runIndex + 2}`).join(',');
  const size = `TEXT(AVERAGE(${references('I')}),"0.00000")`;
  summary.getRange(`B${row}`).formulas = [[cohort.seeds > 1
    ? `=${size}&" ± "&TEXT(STDEV(${references('I')}),"0.00000")` : `=${size}`]];
  for (const [target, input, operation, value] of [
    ['C', 'J', 'AVERAGE', cohort.delta_base], ['D', 'K', 'AVERAGE', cohort.mean_solve_time_s],
    ['E', 'L', 'AVERAGE', cohort.feasible_rate], ['G', 'N', 'MAX', cohort.peak_memory_mib],
    ['J', 'Q', 'AVERAGE', cohort.base_gain], ['K', 'R', 'AVERAGE', cohort.rule_gain],
    ['L', 'S', 'AVERAGE', cohort.mean_search_steps], ['M', 'X', 'SUM', cohort.wins],
    ['N', 'Y', 'SUM', cohort.ties], ['O', 'Z', 'SUM', cohort.losses],
    ['P', 'AA', 'SUM', cohort.deadline_overruns], ['Q', 'AB', 'MAX', cohort.max_deadline_overrun_s],
  ]) {
    if (value !== null) summary.getRange(`${target}${row}`).formulas = [[`=${operation}(${references(input)})`]];
  }
}

const notes = [
  ['统计口径', 'outputs/v4_frequency/summary.json', '样本', '已完成方法run', report.runs.length],
  ['统计口径', 'outputs/v4_frequency/summary.json', '样本', '已完成配置组', report.cohorts.length],
  ['统计口径', '', 'Mean Size', '均值与标准差', '逐骨干种子图均值的均值与样本标准差；单种子不报告标准差'],
  ['统计口径', '', 'Δ原骨干', '参考', '相同划分、精度、协议、后处理及种子的 baseline；无匹配参考时留空'],
  ['统计口径', '', '配对', '胜平负', '按同图同种子比较；共享初始解与独立搜索分别标注；合并行计图—种子记录'],
  ['统计口径', '', '后处理', '后处理增益', '同运行 final_size − base_size；平台增益为 final_size − rule_size'],
  ['统计口径', '', 'PAVE', '冻结评分器', '使用对应骨干种子的 v3 stats_h1 评分器；与无后处理分行展示'],
  ['统计口径', '', '显存', '峰值口径', 'PyTorch峰值已分配显存；不含CUDA上下文及allocator保留缓存'],
  ['统计口径', '', '时间', '计时口径', '正式逐图solve_time_s包含搜索与后处理；机制诊断计时不进入主表'],
  ['统计口径', '', '筛选', '划分与精度', 'validation、test、FP32、FP16、固定量及限时结果分别汇总'],
];
const diagnostics = [...notes, ...report.diagnostics];
const mechanisms = writeSheet('V4机制', ['任务类型', '来源', '阶段或比较', '指标', '数值'], diagnostics,
  [22, 64, 64, 40, 80], { E: '0.00000' });
for (const [index, row] of diagnostics.entries()) {
  if (typeof row[4] === 'number' && Number.isInteger(row[4]) &&
      !/增益|Size|差|平均|秒|准确率|比例|Regret|regret|后悔|误差|lr/.test(row[3])) {
    mechanisms.getRange(`E${index + 2}`).setNumberFormat('#,##0');
  }
}
book.recalculate();

for (const previous of preserved) {
  const current = book.worksheets.getItem(previous.name).getUsedRange();
  if (JSON.stringify(current?.values) !== previous.values || JSON.stringify(current?.formulas) !== previous.formulas) {
    throw new Error(`Existing worksheet values or formulas changed: ${previous.name}`);
  }
}
const overview = await book.inspect({ kind: 'table', range: 'V4汇总!A1:I7', include: 'values,formulas',
  tableMaxRows: 7, tableMaxCols: 9, maxChars: 4500 });
console.log(overview.ndjson);
for (const sheetName of names) {
  const errors = await book.inspect({ kind: 'match', sheetId: sheetName,
    searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',
    options: { useRegex: true, maxResults: 20 }, maxChars: 1600 });
  console.log(errors.ndjson);
  const lastColumn = sheetName === 'V4机制' ? 'E' : 'I';
  const count = sheetName === 'V4机制' ? diagnostics.length : sheetName === 'V4汇总' ? report.cohorts.length : report.runs.length;
  const preview = await book.render({ sheetName, range: `A1:${lastColumn}${Math.min(count + 1, 12)}`, scale: 1 });
  await fs.writeFile(path.join(scratch, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const updated = await SpreadsheetFile.exportXlsx(book);
const staged = path.join(scratch, '实验结果.xlsx');
await updated.save(staged);
await fs.rename(staged, destination);
console.log(`Updated V4 sheets: ${report.cohorts.length} configurations, ${report.runs.length} runs, ${diagnostics.length} diagnostics`);
