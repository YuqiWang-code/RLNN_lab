import fs from 'node:fs/promises';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';

const library = process.env.RUNTIME_NODE_MODULES
  ? pathToFileURL(path.join(process.env.RUNTIME_NODE_MODULES, '@oai/artifact-tool/dist/artifact_tool.mjs')).href
  : '@oai/artifact-tool';
const { Presentation, PresentationFile } = await import(library);
const directory = path.resolve(process.argv[2] || 'tmp/model-figure-v5');
await fs.mkdir(directory, { recursive: true });

const presentation = Presentation.create({ slideSize: { width: 1140, height: 449 } });
const slide = presentation.slides.add();
const c = {
  white: '#FFFFFF', ink: '#242525', outline: '#515456', muted: '#424B55',
  blue: '#C8D7EF', strongBlue: '#355B9B', green: '#DFEBCD',
  leaf: '#78A64D', forest: '#3C6B35', gold: '#FBE6A4',
  apricot: '#F8D3AB', cream: '#F8F2E9',
};
slide.background.fill = c.white;

function shape(name, x, y, w, h, fill = 'none', stroke = 'none', pen = 0, geometry = 'rect') {
  return slide.shapes.add({
    name, geometry, position: { left: x, top: y, width: w, height: h },
    fill, line: { style: 'solid', fill: stroke, width: pen },
  });
}

function label(name, x, y, w, h, content, size, options = {}) {
  const box = shape(name, x, y, w, h, options.fill || 'none', 'none', 0, 'textbox');
  box.text = content;
  box.text.style = {
    typeface: 'Times New Roman', fontSize: size, bold: !!options.bold, italic: !!options.italic,
    color: options.color || c.ink, alignment: options.align || 'left', verticalAlignment: 'middle',
    wrap: 'none', autoFit: 'none', insets: 0,
  };
  return box;
}

function connect(a, b, fromSide, toSide, color = c.ink, arrow = true, width = 1.8) {
  return slide.shapes.connect(a, b, {
    kind: 'straight', fromSide, toSide, line: { style: 'solid', fill: color, width },
    ...(arrow ? { tail: { type: 'triangle', width: 'med', length: 'med' } } : {}),
  });
}

function dot(name, x, y, radius, fill, border = fill, pen = 1) {
  return shape(name, x - radius, y - radius, 2 * radius, 2 * radius, fill, border, pen, 'ellipse');
}

function graph(name, x, y) {
  const xy = [[-22, 6], [-11, -14], [11, -14], [22, 6], [0, 15]];
  const nodes = xy.map(([dx, dy], i) => {
    const selected = [0, 2].includes(i);
    return dot(name + ' vertex ' + i, x + dx, y + dy, 4.2, selected ? c.leaf : c.white, c.ink, 1.25);
  });
  for (const [a, b] of [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [1, 4]]) {
    connect(nodes[a], nodes[b], undefined, undefined, c.outline, false, 1.3);
  }
  nodes.forEach(node => node.bringToFront());
}

function clock(name, x, y) {
  const center = dot(name + ' face', x, y, 9.1, c.gold, c.ink, 1.25);
  const handY = shape(name + ' minute', x - .65, y - 5.6, 1.3, 5.8, c.ink);
  const handX = shape(name + ' hour', x, y - .6, 4.4, 1.2, c.ink);
  return [center, handX, handY];
}

function ranking(name, x, y) {
  [36, 28, 19, 10].forEach((width, i) => {
    shape(name + ' bar ' + i, x - 18, y - 13 + 8 * i, width, 4.1, i === 0 ? c.strongBlue : c.outline);
  });
}

function exchange(name, x, y) {
  const a = dot(name + ' removed', x, y - 10, 4.5, c.white, c.ink, 1.25);
  const b = dot(name + ' added left', x - 15, y + 12, 4.5, c.leaf, c.ink, 1.25);
  const d = dot(name + ' added right', x + 15, y + 12, 4.5, c.leaf, c.ink, 1.25);
  [b, d].forEach(end => connect(a, end, undefined, undefined, c.outline, false, 1.5));
  [a, b, d].forEach(node => node.bringToFront());
}

function particles(name, x, y) {
  const pattern = ['1010', '0101', '1001'];
  pattern.forEach((row, j) => [...row].forEach((v, i) => {
    shape(name + ' bit ' + j + '-' + i, x - 18 + i * 9, y - 12 + j * 9, 7.2, 7.2,
      v === '1' ? c.strongBlue : c.white, c.outline, .65);
  }));
}

function decoded(name, x, y) {
  for (let i = 0; i < 4; i++)
    shape(name + ' state ' + i, x - 20 + i * 3.4, y - 16 + i * 2.8, 29, 24, i === 3 ? c.white : c.gold, c.outline, 1.0);
  const a = dot(name + ' free end left', x - 5.5, y + 8, 2.5, c.leaf, c.ink, .7);
  const b = dot(name + ' excluded center', x + 4.0, y + .2, 2.35, c.white, c.ink, .7);
  const d = dot(name + ' free end right', x + 13.0, y + 8, 2.5, c.leaf, c.ink, .7);
  [a, d].forEach(node => connect(b, node, undefined, undefined, c.outline, false, .85));
  [a, b, d].forEach(node => node.bringToFront());
}

function stage(name, x, y, w, h, fill, heading, subline, icon) {
  const body = shape(name + ' body', x, y, w, h, fill, c.outline, 1.35, 'roundRect');
  if (icon) icon(name, x + 37, y + h / 2);
  const tx = x + (icon ? 72 : 10);
  const tw = x + w - tx - 6;
  label(name + ' title', tx, y + (subline ? 3 : 0), tw, subline ? 26 : h, heading, subline ? 22 : 20,
    { bold: true, align: icon ? 'left' : 'center' });
  if (subline) label(name + ' detail', tx, y + 26, tw, 22, subline, 17.4, { color: c.muted });
  return body;
}

function frame(name, x, title, titleWidth, timeSource) {
  shape(name + ' frame', x, 20, 552, 319, c.white, c.outline, 1.45, 'roundRect');
  label(name + ' heading', x + (552 - titleWidth) / 2, 3, titleWidth, 32, title, 25.5,
    { bold: true, align: 'center', fill: c.white });
  clock(name + ' clock', x + 34, 52);
  label(name + ' clock start', x + 52, 39, 484, 26, timeSource, 19, { italic: true });
}

frame('Feasible', 8, '(a)  Feasible search', 224, 'Clock starts at the host edge list');
frame('Particle', 580, '(b)  Particle search', 224, 'Clock starts at per-graph buffer load and reset');

const initial = stage('Greedy seed', 25, 77, 331, 52, c.green, 'Greedy initial set', 'Minimum remaining degree', graph);
const rank = stage('Ranking', 25, 171, 331, 52, c.blue, 'Rank vertices once', 'Degree + GCN, or degree alone', ranking);
const refine = stage('Refinement', 25, 270, 331, 52, c.green, 'Feasible refinement', 'Exchanges and RB clique bounds', exchange);
connect(initial, rank, 'bottom', 'top');
connect(rank, refine, 'bottom', 'top');

const flips = stage('GPU replay', 597, 73, 331, 49, c.blue, 'Particle flip policy', '200 particles on the GPU', particles);
const midpoint = stage('Midpoint decode', 597, 143, 331, 51, c.gold, 'Midpoint decode', 'A.3 on four archive states', decoded);
const replay = stage('More replay', 597, 214, 331, 35, c.blue, 'Continue 10-step replay blocks', '', null);
const terminal = stage('Terminal decode', 597, 269, 331, 53, c.apricot, 'Terminal decode', 'Pending archive, if time remains', decoded);
connect(flips, midpoint, 'bottom', 'top');
connect(midpoint, replay, 'bottom', 'top');
connect(replay, terminal, 'bottom', 'top');

shape('Shared host archive frame', 8, 360, 1124, 78, c.cream, c.outline, 1.45, 'roundRect');
for (const [x, name] of [[324, 'cutoff'], [801, 'return']])
  shape('Host separator ' + name, x, 373, 1, 52, '#CABCAA');

function archiveIcon(x, y) {
  [0, 1, 2].forEach(i => shape('Host stack layer ' + i, x + 3 * i, y - 3 * i, 31, 21,
    [c.gold, c.green, c.white][i], c.outline, 1.1));
  shape('Host stack record', x + 11, y + 2, 17, 2.3, c.forest);
  shape('Host stack timestamp', x + 11, y + 8, 10, 1.7, c.outline);
}
archiveIcon(30, 393);
label('Host archive title', 85, 368, 230, 28, 'Host archive', 23.5, { bold: true });
label('Host archive detail', 85, 399, 229, 27, 'Timestamp each feasible set.', 18);
label('Cutoff title', 349, 368, 422, 28, 'At cutoff B', 23.5, { bold: true });
label('Cutoff detail', 349, 399, 441, 27, 'Score the largest set copied by the deadline.', 18);
label('Return title', 827, 368, 286, 28, 'At return T', 23.5, { bold: true });
label('Return detail', 827, 399, 286, 27, 'Record elapsed wall time.', 18);

function publish(name, x, joinX, source, atY, heading, detail) {
  const end = dot(name + ' join', joinX, atY, 2.8, c.forest);
  connect(source, end, 'right', 'left', c.forest, true, 1.7);
  label(name + ' label', x, atY - 28, joinX - x - 6, 24, heading, 18.2, { bold: true, align: 'center', color: c.forest });
  label(name + ' detail', x, atY + 5, joinX - x - 6, 20, detail, 16.6, { italic: true, align: 'center', color: c.muted });
  return end;
}

function deliver(name, first, last, x) {
  connect(first, last, 'bottom', 'top', c.forest, false, 1.8);
  const destination = shape(name + ' host inlet', x - .15, 359.9, .3, .2);
  connect(last, destination, 'bottom', 'top', c.forest, true, 2.0);
}

const seedOut = publish('Greedy publication', 358, 536, initial, 103, 'Publish initial set', 'Before ranking');
const improvedOut = publish('Refinement publication', 358, 536, refine, 296, 'Publish larger sets', 'On improvement');
deliver('Feasible publication rail', seedOut, improvedOut, 536);
const middleOut = publish('Midpoint publication', 930, 1108, midpoint, 168.5, 'Publish decoded set', 'Before search resumes');
const finalOut = publish('Terminal publication', 930, 1108, terminal, 295.5, 'Keep the larger set', 'Within the same budget');
deliver('Particle publication rail', middleOut, finalOut, 1108);

slide.speakerNotes.text = 'The dark arrows in each panel give the chronological sequence. Each green path carries an actual feasible vertex set to the host archive and records its arrival. The two timers start at the respective graph inputs. On the particle path, A.3 decodes the four lowest-energy states of the archive at each readout when time permits. Both readouts and host transfers are charged to the same instance budget. The host deadline score may be available before the solver returns.';
await (await PresentationFile.exportPptx(presentation)).save(path.join(directory, 'candidate.pptx'));
const groupingScript = fileURLToPath(new URL('./group_search_paths.py', import.meta.url));
execFileSync(process.env.PYTHON || 'python', [groupingScript, directory], { stdio: 'inherit' });
