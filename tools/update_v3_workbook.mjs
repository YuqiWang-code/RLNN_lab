#!/usr/bin/env node
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const scratch = path.join(root, 'tmp', 'v3_reporting');
const packages = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
await fs.mkdir(scratch, { recursive: true });
try {
  await fs.symlink(packages, path.join(scratch, 'node_modules'), process.platform === 'win32' ? 'junction' : 'dir');
} catch (error) {
  if (error.code !== 'EEXIST') throw error;
}
const require = createRequire(path.join(scratch, 'package.json'));
const { SpreadsheetFile, FileBlob } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const source = path.join(root, 'outputs', 'v3', 'summary.json');
const destination = path.join(root, 'docs', '实验结果.xlsx');
const report = JSON.parse(await fs.readFile(source, 'utf8'));
const book = await SpreadsheetFile.importXlsx(await FileBlob.load(destination));
const preserved = book.worksheets.items.filter(sheet => !['V3汇总', 'V3逐种子', 'V3机制'].includes(sheet.name))
  .map(sheet => ({ name: sheet.name,
    values: JSON.stringify(sheet.getUsedRange()?.values), formulas: JSON.stringify(sheet.getUsedRange()?.formulas) }));

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

writeSheet('V3逐种子', ['方法', '划分', '协议', '骨干种子', '评分种子', '搜索种子', '图数',
  'Mean Size', 'ΔRLNN', 'Δ规则', '平均求解 (s)', '可行率', 'Params', '可训练参数', '峰值显存 (MiB)',
  '预算 (s)', 'H', '候选上限', '外层轮数', '胜', '平', '负', '同运行后处理增益', '同运行平台增益',
  '平均搜索步数', '中性决策/图', '评分秒/图', '后处理秒/图', '超时图数', '来源'],
report.runs.map(run => [run.method, run.split, run.protocol, run.backbone_seed, run.head_seed,
  run.search_seed, run.samples, run.mean_size, run.delta_base, run.delta_rule, run.mean_solve_time_s,
  run.feasible_rate, run.params, run.trainable_params, run.peak_memory_mib, run.time_budget_s, run.horizon,
  run.candidate_cap, run.outer_rounds, run.wins, run.ties, run.losses, run.base_gain, run.rule_gain,
  run.search_steps, run.neutral_decisions, run.score_time_s, run.post_time_s, run.deadline_overruns, run.source]),
[36, 13, 18, 11, 11, 11, 10, 14, 13, 13, 16, 12, 14, 14, 18, 11, 8, 11, 11, 10, 10, 10, 22, 22, 17, 17, 17, 17, 13, 65],
{ H: '0.00000', I: '+0.00000;-0.00000;0.00000', J: '+0.00000;-0.00000;0.00000', K: '0.00000',
  L: '0.00%', M: '#,##0', N: '#,##0', O: '0.00', P: '0.00', W: '0.00000', X: '0.00000',
  Y: '0.000', Z: '0.000', AA: '0.000000', AB: '0.000000' });

const summary = writeSheet('V3汇总', ['方法', 'Mean Size ↑（均值±标准差）', 'ΔRLNN ↑', 'Δ规则 ↑',
  '平均求解 (s) ↓', '可行率 ↑', 'Params', '可训练参数', '峰值显存 (MiB)', '协议', '预算 (s)',
  '种子数', '图数/种子', '胜', '平', '负', '同运行后处理增益', '同运行平台增益', '超时图数', 'H', '候选上限', '外层轮数', '划分'],
report.cohorts.map(cohort => [cohort.method, null, cohort.delta_base, cohort.delta_rule,
  cohort.mean_solve_time_s, cohort.feasible_rate, cohort.params, cohort.trainable_params,
  cohort.peak_memory_mib, cohort.protocol, cohort.time_budget_s, cohort.seeds, cohort.samples,
  cohort.wins, cohort.ties, cohort.losses, cohort.base_gain, cohort.rule_gain, cohort.deadline_overruns,
  cohort.horizon, cohort.candidate_cap, cohort.outer_rounds, cohort.split]),
[36, 30, 15, 15, 19, 13, 14, 14, 18, 18, 11, 10, 13, 11, 11, 11, 22, 22, 13, 8, 11, 11, 13],
{ C: '+0.00000;-0.00000;0.00000', D: '+0.00000;-0.00000;0.00000', E: '0.00000', F: '0.00%',
  G: '#,##0', H: '#,##0', I: '0.00', K: '0.00', Q: '0.00000', R: '0.00000' });

for (const [index, cohort] of report.cohorts.entries()) {
  const row = index + 2;
  const references = letter => cohort.run_indices.map(runIndex => `'V3逐种子'!${letter}${runIndex + 2}`).join(',');
  const size = `TEXT(AVERAGE(${references('H')}),"0.00000")`;
  summary.getRange(`B${row}`).formulas = [[cohort.seeds > 1
    ? `=${size}&" ± "&TEXT(STDEV(${references('H')}),"0.00000")` : `=${size}`]];
  for (const [target, input, operation, value] of [
    ['C', 'I', 'AVERAGE', cohort.delta_base], ['D', 'J', 'AVERAGE', cohort.delta_rule],
    ['E', 'K', 'AVERAGE', cohort.mean_solve_time_s], ['F', 'L', 'AVERAGE', cohort.feasible_rate],
    ['I', 'O', 'MAX', cohort.peak_memory_mib], ['N', 'T', 'SUM', cohort.wins],
    ['O', 'U', 'SUM', cohort.ties], ['P', 'V', 'SUM', cohort.losses],
    ['Q', 'W', 'AVERAGE', cohort.base_gain], ['R', 'X', 'AVERAGE', cohort.rule_gain],
    ['S', 'AC', 'SUM', cohort.deadline_overruns],
  ]) {
    if (value !== null) summary.getRange(`${target}${row}`).formulas = [[`=${operation}(${references(input)})`]];
  }
}

writeSheet('V3机制', ['任务类型', '来源', '邻域或阶段', '指标', '数值'], report.diagnostics,
  [22, 64, 35, 42, 60], { E: '0.00000' });
book.recalculate();

for (const previous of preserved) {
  const current = book.worksheets.getItem(previous.name).getUsedRange();
  if (JSON.stringify(current?.values) !== previous.values || JSON.stringify(current?.formulas) !== previous.formulas) {
    throw new Error(`Existing worksheet values or formulas changed: ${previous.name}`);
  }
}
const overview = await book.inspect({ kind: 'table', range: 'V3汇总!A1:I7', include: 'values,formulas',
  tableMaxRows: 7, tableMaxCols: 9, maxChars: 4500 });
console.log(overview.ndjson);
for (const sheetName of ['V3汇总', 'V3逐种子', 'V3机制']) {
  const errors = await book.inspect({ kind: 'match', sheetId: sheetName,
    searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',
    options: { useRegex: true, maxResults: 20 }, maxChars: 1600 });
  console.log(errors.ndjson);
  const lastColumn = sheetName === 'V3机制' ? 'E' : 'I';
  const count = sheetName === 'V3机制' ? report.diagnostics.length : sheetName === 'V3汇总' ? report.cohorts.length : report.runs.length;
  const preview = await book.render({ sheetName, range: `A1:${lastColumn}${Math.min(count + 1, 12)}`, scale: 1 });
  await fs.writeFile(path.join(scratch, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
const updated = await SpreadsheetFile.exportXlsx(book);
const staged = path.join(scratch, '实验结果.xlsx');
await updated.save(staged);
await fs.rename(staged, destination);
console.log(`Updated V3 sheets: ${report.cohorts.length} cohorts, ${report.runs.length} runs, ${report.diagnostics.length} diagnostics`);
