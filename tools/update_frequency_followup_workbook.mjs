#!/usr/bin/env node
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const analysis = path.join(root, 'outputs/v4_frequency/followup/analysis');
const scratch = path.join(root, 'tmp/frequency_followup_reporting');
const destination = path.join(root, 'docs/实验结果.xlsx');
const packages = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules');
const names = ['频域追诊总览', '频域追诊P1', '频域追诊P2', '频域追诊P3', '频域追诊P4'];

function parseCsv(text) {
  const rows = [];
  let row = [], value = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { value += '"'; i++; }
      else quoted = !quoted;
    } else if (char === ',' && !quoted) { row.push(value); value = ''; }
    else if (char === '\n' && !quoted) {
      row.push(value.replace(/\r$/, '')); rows.push(row); row = []; value = '';
    } else value += char;
  }
  if (value || row.length) { row.push(value.replace(/\r$/, '')); rows.push(row); }
  const headers = (rows.shift() ?? []).map(value => value.replace(/^\uFEFF/, ''));
  return rows.filter(row => row.some(value => value !== '')).map(row => Object.fromEntries(headers.map((key, i) => [key, typed(row[i] ?? '')])));
}

function typed(value) {
  if (value === '' || value === 'NaN' || value === 'nan') return null;
  if (value === 'True' || value === 'true') return true;
  if (value === 'False' || value === 'false') return false;
  if (/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) return Number(value);
  return value;
}

async function readCsv(filename) {
  try { return parseCsv(await fs.readFile(path.join(analysis, filename), 'utf8')); }
  catch (error) { if (error.code === 'ENOENT') return []; throw error; }
}

const labels = {
  study: '实验口径', split: '划分', model: '模型', trajectory: '状态轨迹', trajectory_model: '状态轨迹',
  variant: '干预或分支', precision: '精度', step: '时刻', signal: '信号', seed: '训练种子', R: '重复数R',
  candidate_sampling: '候选采样', candidate_group: '候选类别', category: '节点类别', group: '节点类别', horizon: '续跑步数H',
  metric: '配对指标', method: '方法', reference: '参考', post: '后处理', component: '组件', unit: '计时单位',
  mean_delta: '平均差值', graph_ci_low: '图级CI下界', graph_ci_high: '图级CI上界', wins: '胜', ties: '平', losses: '负',
  positive_seeds: '正增益种子数', seed_count: '种子数', pairs: '图—种子对数', by_seed: '分种子差值',
  regret_improvement_over_MSG: 'MSG−当前后悔值', base_size: '初始Size', final_size: '最终Size', pave_gain: 'PAVE增益',
  mean_size: 'Mean Size', std_size: '种子SD', mean_solve_time_s: '平均求解 (s)', feasible_rate: '可行率',
  params: 'Params', peak_memory_mib: 'PyTorch峰值已分配显存 (MiB)', mean_search_steps: '平均搜索步数',
  protocol: '协议', extended_search: '持续搜索', time_budget_s: '预算 (s)', deadline_overruns: '超时数',
  raw_low_fraction: '原始低频比例', raw_mid_fraction: '原始中频比例', raw_higher_fraction: '原始较高频比例',
  raw_extreme_fraction: '原始极高频比例', centered_low_fraction: '去驻留低频比例', centered_mid_fraction: '去驻留中频比例',
  centered_higher_fraction: '去驻留较高频比例', centered_extreme_fraction: '去驻留极高频比例',
  stationary_fraction: '驻留能量比例', total_energy: '总谱能量', nonstationary_energy: '非驻留谱能量',
  energy_per_column: '每通道谱能量', eigenvalue_max: '最大特征值',
  delta_probability_abs_mean: '平均|Δp|', delta_probability_mean: '平均Δp', delta_logit_rms: 'Δlogit RMS',
  expected_flips: '期望翻转数', actual_flips: '实际翻转数', flips_target_error: '翻转目标误差', top5_jaccard: 'Top5 Jaccard',
  probability_below_001: 'p<0.01比例', probability_above_099: 'p>0.99比例',
  expected_sum_single_flip_delta_energy: '概率加权单翻转ΔE', favorable_probability_mass: '有利翻转概率质量',
  logit_immediate_gain_correlation: 'logit与即时收益相关',
  pair_accuracy: '状态内排序准确率', pooled_pair_accuracy: '候选对加权准确率', top1_gain: 'Top1收益',
  top1_regret: 'Top1后悔值', uniform_gain: '均匀候选收益', oracle_sample_gain: '标签内Oracle收益',
  top1_positive: 'Top1正收益比例', states: '状态数', informative_states: '信息状态数',
  non_tied_pairs: '非平局候选对', correct_pair_credit: '正确排序计分',
  informative: '信息状态比例', zero_replicate_fraction: '逐次零收益比例', mean_gain_sem: '候选收益标准误',
  half_spearman: '前后半重复Spearman', half_top1_same: '前后半Top1一致率', half_top1_set_jaccard: '前后半Top1集合Jaccard',
  half_action_sign_agreement: '前后半动作符号一致率', half_pair_direction_agreement: '前后半候选对同向率',
  pooled_half_direction_agreement: '候选对合并同向率', half_joint_non_tied_pairs: '前后半均非平局对',
  half_direction_match_pairs: '前后半同向对', positive_mean_fraction: '平均收益为正的候选比例',
  time_s: '耗时 (s)', solve_time_s: '求解时间 (s)', search_time_s: '搜索时间 (s)', post_time_s: '后处理时间 (s)',
  search_steps: '搜索步数', median_s: '耗时中位数 (s)', p95_s: '耗时P95 (s)', mad_s: '耗时MAD (s)',
};

function label(key) {
  if (labels[key]) return labels[key];
  for (const suffix of ['_mean', '_std']) {
    if (key.endsWith(suffix)) return `${label(key.slice(0, -suffix.length))} ${suffix === '_std' ? '种子SD' : '均值'}`;
  }
  return key;
}

function column(index) {
  let name = '';
  for (let n = index + 1; n; n = Math.floor((n - 1) / 26)) name = String.fromCharCode(65 + (n - 1) % 26) + name;
  return name;
}

function priority(keys, first) { return [...first.filter(key => keys.includes(key)), ...keys.filter(key => !first.includes(key))]; }

const plans = [
  { sheet: names[1], title: 'P1 编码阶段频谱：图内平均，再按图与种子汇总', file: 'stage_summary.csv',
    first: ['split', 'model', 'trajectory', 'variant', 'step', 'signal', 'stationary_fraction_mean', 'raw_higher_fraction_mean', 'centered_higher_fraction_mean'] },
  { sheet: names[1], title: 'P1 节点类别内门控、残差与行为变化', file: 'spectral_behavior_summary.csv',
    first: ['split', 'model', 'trajectory', 'variant', 'group', 'step'] },
  { sheet: names[2], title: 'P2 共享初始状态与随机流的完整replay', file: 'replay_summary.csv',
    first: ['split', 'precision', 'variant', 'base_size_mean', 'base_size_std', 'final_size_mean', 'final_size_std', 'pave_gain_mean'] },
  { sheet: names[2], title: 'P2 相对Full逐图配对；CI以图ID为重采样单位', file: 'replay_paired.csv',
    first: ['split', 'precision', 'metric', 'method', 'reference', 'mean_delta', 'graph_ci_low', 'graph_ci_high', 'wins', 'ties', 'losses'] },
  { sheet: names[2], title: 'P2 同状态行为变化', file: 'same_state_summary.csv',
    first: ['split', 'precision', 'variant', 'category', 'step', 'delta_probability_abs_mean_mean', 'expected_flips_mean', 'delta_logit_rms_mean'] },
  { sheet: names[2], title: 'P2 同模型重复回放的轨迹与 Size 波动', file: 'native_repeat_summary.csv',
    select: ['precision', 'reference_variant', 'graph_count', 'seed_count', 'paired_repeat_count', 'changed_observation_phases_rate_mean', 'selected_nodes_changed_mean', 'base_size_delta_changed_mean', 'final_size_delta_changed_mean', 'base_size_delta_mean', 'base_size_delta_abs_mean', 'final_size_delta_abs_mean'] },
  { sheet: names[3], title: 'P3 成对翻转收益排序：共享标签，按图汇总', file: 'utility_summary.csv',
    first: ['study', 'split', 'model', 'R', 'candidate_group', 'pair_accuracy_mean', 'pair_accuracy_std', 'top1_gain_mean', 'top1_regret_mean', 'pooled_pair_accuracy'] },
  { sheet: names[3], title: 'P3 相对MSG的图级后悔值配对；正改善值表示当前模型更好', file: 'utility_paired.csv',
    first: ['study', 'split', 'R', 'candidate_group', 'method', 'reference', 'regret_improvement_over_MSG', 'graph_ci_low', 'graph_ci_high', 'positive_seeds'] },
  { sheet: names[3], title: 'P3 标签重复稳定性：前后半独立续跑', file: 'reliability_summary.csv',
    first: ['study', 'split', 'R', 'candidate_group', 'informative_mean', 'mean_gain_sem_mean', 'half_spearman_mean', 'half_top1_same_mean', 'pooled_half_direction_agreement'] },
  { sheet: names[4], title: 'P4 真实端到端重复测量', file: 'profile_e2e.csv',
    first: ['split', 'precision', 'model', 'post', 'final_size_mean', 'final_size_std', 'solve_time_s_mean', 'median_s', 'p95_s', 'mad_s', 'search_steps_mean'] },
  { sheet: names[4], title: 'P4 组件独立计时；不可直接相加替代端到端时间', file: 'profile_components.csv',
    first: ['split', 'precision', 'model', 'component', 'unit', 'time_s_mean', 'time_s_std'] },
  { sheet: names[4], title: 'P4 固定输入的融合一致性与原生前向波动', file: 'component_equivalence_summary.csv',
    select: ['split', 'precision', 'model', 'precision_cached_fusion_max_abs_max', 'fp32_decomposition_vs_native_max_abs_max', 'native_pairwise_max_abs_max', 'native_probability_max_abs_max', 'decomposition_vs_native_max_abs_max'] },
  { sheet: names[4], title: '既有V4验证质量与时间参考', file: 'historical_quality_time.csv',
    select: ['method', 'mean_size', 'std_size', 'mean_solve_time_s', 'feasible_rate', 'params', 'peak_memory_mib', 'mean_search_steps', 'protocol', 'precision', 'split', 'time_budget_s', 'extended_search', 'deadline_overruns'] },
];

const loaded = await Promise.all(plans.map(async plan => ({ ...plan, rows: await readCsv(plan.file) })));
const sections = loaded.filter(plan => plan.rows.length);
if (!sections.some(plan => plan.file !== 'historical_quality_time.csv')) throw new Error('No completed followup result tables; workbook was not edited.');

await fs.mkdir(scratch, { recursive: true });
try { await fs.symlink(packages, path.join(scratch, 'node_modules'), process.platform === 'win32' ? 'junction' : 'dir'); }
catch (error) { if (error.code !== 'EEXIST') throw error; }
const require = createRequire(path.join(scratch, 'package.json'));
const { SpreadsheetFile, FileBlob } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const book = await SpreadsheetFile.importXlsx(await FileBlob.load(destination));
const preserved = book.worksheets.items.filter(sheet => !names.includes(sheet.name)).map(sheet => ({
  name: sheet.name, values: JSON.stringify(sheet.getUsedRange()?.values), formulas: JSON.stringify(sheet.getUsedRange()?.formulas),
}));
const overview = [
  { kind: '统计口径', scope: '均值与标准差', metric: '聚合单位', value: '状态/粒子先在图内平均，再对图平均；SD为训练种子均值的样本标准差', source: 'tools/summarize_frequency_followup.py' },
  { kind: '统计口径', scope: '数据划分', metric: '解释边界', value: 'train用于机制探索；validation已参与选权重，不能作为独立测试证明', source: '' },
  { kind: '统计口径', scope: 'P2配对', metric: '置信区间', value: '以图ID为bootstrap单位，同一图的不同种子共同保留；差值为当前方法−Full', source: '' },
  { kind: '统计口径', scope: 'P3收益', metric: '标签', value: '同状态、同候选、冻结B续跑标签；R2/4/8/16是同一最大R的前缀，不是独立实验', source: '' },
  { kind: '统计口径', scope: 'P3采样', metric: '分母', value: 'Uniform与Stratified分开；release_high可与五种状态类别重叠；重复不是独立图', source: '' },
  { kind: '统计口径', scope: 'P3配对', metric: '后悔值方向', value: 'mean_delta/CI为当前−MSG后悔值；regret_improvement_over_MSG取相反数，正值表示改善', source: '' },
  { kind: '统计口径', scope: 'P4计时', metric: '边界', value: '组件微基准与实际端到端计时分开；诊断墙钟不进入正式求解时间', source: '' },
];
for (const section of sections) overview.push({ kind: '数据来源', scope: section.sheet, metric: section.title, value: section.rows.length, source: `outputs/v4_frequency/followup/analysis/${section.file}` });
try {
  const decision = JSON.parse(await fs.readFile(path.join(analysis, 'decision.json'), 'utf8'));
  function flatten(value, prefix = '') {
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      for (const [key, child] of Object.entries(value)) flatten(child, prefix ? `${prefix}.${key}` : key);
    } else overview.push({ kind: '阶段决策', scope: '机制证据', metric: prefix, value: Array.isArray(value) ? JSON.stringify(value) : value, source: 'outputs/v4_frequency/followup/analysis/decision.json' });
  }
  flatten(decision);
} catch (error) { if (error.code !== 'ENOENT') throw error; }
sections.unshift({ sheet: names[0], title: '频域机制追诊：统计口径与证据范围', file: 'decision.json', rows: overview,
  select: ['kind', 'scope', 'metric', 'value', 'source'] });
Object.assign(labels, { kind: '类别', scope: '范围', metric: '指标', value: '数值或说明', source: '来源' });

const written = [];
for (const name of names) {
  const parts = sections.filter(section => section.sheet === name);
  if (!parts.length) continue;
  const sheet = book.worksheets.items.find(item => item.name === name) ?? book.worksheets.add(name);
  sheet.getUsedRange()?.unmerge();
  sheet.getUsedRange()?.clear({ applyTo: 'contents' });
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(2);
  let row = 1;
  const widths = [];
  const blocks = [];
  for (const part of parts) {
    const available = Object.keys(part.rows[0]);
    const keys = part.select ? part.select.filter(key => available.includes(key)) : priority(available, part.first ?? []);
    const last = column(keys.length - 1);
    const headerRow = row + 1, bottom = headerRow + part.rows.length;
    sheet.getRange(`A${row}:${last}${row}`).merge();
    sheet.getRange(`A${row}`).values = [[part.title]];
    sheet.getRange(`A${row}:${last}${row}`).format.fill = '#EAF0F6';
    sheet.getRange(`A${row}`).format.font.bold = true;
    sheet.getRange(`A${row}`).format.font.size = 12;
    const range = sheet.getRange(`A${headerRow}:${last}${bottom}`);
    range.values = [keys.map(label), ...part.rows.map(record => keys.map(key => record[key] ?? null))];
    range.format.font.name = 'Arial';
    range.format.font.size = 10;
    range.format.rowHeight = 20;
    range.format.verticalAlignment = 'center';
    const header = sheet.getRange(`A${headerRow}:${last}${headerRow}`);
    header.format.fill = '#243B53';
    header.format.font.color = '#FFFFFF';
    header.format.font.bold = true;
    header.format.rowHeight = 36;
    header.format.wrapText = true;
    for (const [index, key] of keys.entries()) {
      const col = column(index);
      const width = ['study', 'signal', 'variant', 'candidate_group', 'category', 'component', 'metric'].includes(key) ? 32
        : ['value', 'source', 'method'].includes(key) ? 62 : ['model', 'reference', 'precision', 'split', 'R', 'seed'].includes(key) ? 12 : 22;
      widths[index] = Math.max(widths[index] ?? 0, width);
      if (part.rows.some(record => typeof record[key] === 'number')) {
        const integer = ['seed', 'R', 'step', 'horizon', 'states', 'informative_states', 'wins', 'ties', 'losses', 'positive_seeds', 'seed_count', 'pairs', 'params', 'deadline_overruns', 'half_direction_match_pairs', 'half_joint_non_tied_pairs'].includes(key);
        sheet.getRange(`${col}${headerRow + 1}:${col}${bottom}`).setNumberFormat(integer ? '#,##0' : '0.00000');
      }
    }
    blocks.push({ title: part.title, file: part.file, headerRow, bottom, columns: keys.length, rows: part.rows.length });
    row = bottom + 3;
  }
  for (const [index, width] of widths.entries()) sheet.getRange(`${column(index)}1:${column(index)}${row - 1}`).format.columnWidth = width;
  written.push({ name, blocks });
}
book.recalculate();
for (const previous of preserved) {
  const current = book.worksheets.getItem(previous.name).getUsedRange();
  if (JSON.stringify(current?.values) !== previous.values || JSON.stringify(current?.formulas) !== previous.formulas) throw new Error(`Existing worksheet values or formulas changed: ${previous.name}`);
}
for (const { name, blocks } of written) {
  const errors = await book.inspect({ kind: 'match', sheetId: name,
    searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 20 }, maxChars: 1600 });
  console.log(errors.ndjson);
  for (const block of blocks) {
    const preview = await book.render({ sheetName: name,
      range: `A${block.headerRow - 1}:${column(Math.min(block.columns, 9) - 1)}${Math.min(block.bottom, block.headerRow + 8)}`, scale: 1 });
    await fs.writeFile(path.join(scratch, `${name}_${block.file.replace('.csv', '').replace('.json', '')}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
}
const exported = await SpreadsheetFile.exportXlsx(book);
const staged = path.join(scratch, '实验结果.xlsx');
await exported.save(staged);
await fs.rename(staged, destination);
await fs.writeFile(path.join(scratch, 'written_sections.json'), JSON.stringify(written, null, 2));
console.log(JSON.stringify({ updated_sheets: written.map(entry => entry.name), sections: written.reduce((total, entry) => total + entry.blocks.length, 0) }));
