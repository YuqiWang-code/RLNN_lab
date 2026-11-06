#!/usr/bin/env node
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/package.json'));
const { Workbook, SpreadsheetFile, FileBlob } = await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args[i + 1];
};
if (args.includes('--help')) {
  console.log('node tools/collect_results.mjs [--kind single_gpu|smoke] [--outputs outputs] [--workbook docs/实验结果.xlsx] [--preview tmp/results.png]');
  process.exit(0);
}
const outputs = path.resolve(root, option('--outputs', 'outputs'));
const kind = option('--kind', 'single_gpu');
if (!['single_gpu', 'smoke'].includes(kind)) throw new Error('kind must be single_gpu or smoke');
const destination = path.resolve(root, option('--workbook', kind === 'smoke' ? 'docs/冒烟结果.xlsx' : 'docs/实验结果.xlsx'));
const headers = ['version', 'run', 'task', 'dataset', 'split', 'epoch', 'step', 'metric', 'value', 'unit', 'source', 'recorded_at', 'notes'];

async function findMetrics(dir) {
  const found = [];
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    if (entry.isDirectory() && !['checkpoints', 'wandb'].includes(entry.name)) {
      found.push(...await findMetrics(path.join(dir, entry.name)));
    } else if (entry.isFile() && entry.name === 'metrics.csv') {
      found.push(path.join(dir, entry.name));
    }
  }
  return found.sort();
}

function numeric(value, context) {
  if (value === '' || value == null) return null;
  const result = Number(value);
  if (!Number.isFinite(result)) throw new Error(`${context}: non-finite numeric value ${value}`);
  return result;
}

function unit(metric) {
  if (metric.endsWith('_percent')) return '%';
  if (metric.endsWith('/graph_batch_size')) return 'graphs';
  if (metric.endsWith('_s')) return 's';
  if (metric.endsWith('_mib')) return 'MiB';
  if (metric.endsWith('/updates_per_graph')) return 'updates';
  if (metric.endsWith('_batch')) return 'states';
  if (metric.endsWith('/time') || metric.endsWith('/wall_time')) return 's';
  if (metric.endsWith('/independent_set_size')) return 'nodes';
  return '';
}

let workbook;
try {
  await fs.access(destination);
  workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(destination));
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
  workbook = Workbook.create();
  workbook.worksheets.add('Results');
}
const sheet = workbook.worksheets.getItem('Results');
const old = sheet.getUsedRange()?.values ?? [];
if (old[0]?.some(value => value != null && value !== '') && headers.some((header, i) => old[0][i] !== header)) {
  throw new Error('Results headers differ from the documented schema.');
}
const rows = old.slice(1).filter(row => row[0] != null && row[0] !== '');
const incoming = [];
const replacedSources = new Set();
for (const filename of await findMetrics(outputs)) {
  const relative = path.relative(outputs, filename).split(path.sep);
  if (relative.length < 4) throw new Error(`${filename}: expected outputs/<version>/<run>/metrics/metrics.csv`);
  const [version, run] = relative;
  if (version !== 'v000_rlnn_baseline') continue;
  const metadataPath = path.join(outputs, version, run, 'run.json');
  const metadata = JSON.parse(await fs.readFile(metadataPath, 'utf8'));
  if (metadata.kind !== kind) continue;
  if (metadata.status !== 'completed' || metadata.version !== version || metadata.run !== run) {
    throw new Error(`${metadataPath}: completed run provenance required`);
  }
  const source = path.relative(root, filename).split(path.sep).join('/');
  replacedSources.add(source);
  const csv = await Workbook.fromCSV(await fs.readFile(filename, 'utf8'), { sheetName: 'CSV' });
  const values = csv.worksheets.getItemAt(0).getUsedRange()?.values ?? [];
  if (!values.length) throw new Error(`${source}: empty metrics file`);
  const columns = values[0];
  const recordedAt = (await fs.stat(filename)).mtime.toISOString();
  for (const [index, valuesRow] of values.slice(1).entries()) {
    const row = Object.fromEntries(columns.map((column, i) => [column, valuesRow[i]]));
    for (const metric of columns.filter(column => !['epoch', 'step'].includes(column))) {
      const value = numeric(row[metric], `${source}:${index + 2} ${metric}`);
      if (value === null) continue;
      const split = metric.includes('/') ? metric.split('/')[0] : '';
      const datasetKey = { train: 'training_split', val: 'validation_split', test: 'test_split' }[split];
      const inherited = metadata.resume?.inherited_config && row.epoch != null && row.epoch !== '' && Number(row.epoch) <= metadata.resume.epoch;
      const config = inherited ? metadata.resume.inherited_config : metadata.config;
      const splitPath = config?.[{train:'train_path',val:'valid_path',test:'test_path'}[split]];
      incoming.push([version, run, metadata.task, splitPath ?? metadata[datasetKey] ?? metadata.dataset ?? '', split,
        numeric(row.epoch, source), numeric(row.step, source), metric, value,
        unit(metric), source, recordedAt, JSON.stringify({kind:metadata.kind, mode:metadata.mode, measurement:metadata.measurement, resume:metadata.resume, seed:metadata.seed,
          config, device:inherited ? metadata.resume.inherited_device : metadata.device,
          raw_logs:inherited ? metadata.resume.source_run : `outputs/${version}/${run}/`,
          reference:metadata.reference})]);
    }
  }
}
const result = [...rows.filter(row => !replacedSources.has(row[10])), ...incoming];
for (const table of [...sheet.tables.items]) table.delete();
sheet.getUsedRange()?.clear({ applyTo: 'contents' });
sheet.getRange(`A1:M${result.length + 1}`).values = [headers, ...result];
sheet.showGridLines = false;
sheet.tabColor = '#243B53';
const data = sheet.getRange(`A1:M${Math.max(2, result.length + 1)}`);
data.format.font.name = 'Arial';
data.format.font.size = 10;
data.format.verticalAlignment = 'center';
data.format.rowHeight = 22;
data.format.columnWidth = 17;
sheet.getRange('B:B').format.columnWidth = 30;
sheet.getRange('D:D').format.columnWidth = 48;
sheet.getRange('H:H').format.columnWidth = 30;
sheet.getRange('K:K').format.columnWidth = 64;
sheet.getRange('L:L').format.columnWidth = 29;
sheet.getRange('M:M').format.columnWidth = 36;
const header = sheet.getRange('A1:M1');
header.format.fill = '#243B53';
header.format.font.color = '#FFFFFF';
header.format.font.bold = true;
header.format.horizontalAlignment = 'center';
header.format.rowHeight = 28;
sheet.getRange(`F2:G${Math.max(2, result.length + 1)}`).setNumberFormat('0');
sheet.getRange(`I2:I${Math.max(2, result.length + 1)}`).setNumberFormat('0.000000');
sheet.getRange(`L2:L${Math.max(2, result.length + 1)}`).setNumberFormat('yyyy-mm-dd hh:mm:ss');
sheet.freezePanes.freezeRows(1);
sheet.tables.add(`A1:M${Math.max(2, result.length + 1)}`, true, 'ExperimentResults');
workbook.recalculate();
if (option('--preview', null)) {
  const preview = await workbook.render({ sheetName: 'Results', range: 'A1:M8', scale: 1, format: 'png' });
  const previewPath = path.resolve(root, option('--preview'));
  await fs.mkdir(path.dirname(previewPath), { recursive: true });
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  console.log((await workbook.inspect({ kind: 'table', range: 'Results!A1:M8', tableMaxRows: 8, tableMaxCols: 13, maxChars: 2500 })).ndjson);
}
await fs.mkdir(path.dirname(destination), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(destination);
await fs.rm(`${destination}.inspect.ndjson`, { force: true });
console.log(`${path.relative(root, destination)}: ${result.length} metric records from ${replacedSources.size} CSV files`);
