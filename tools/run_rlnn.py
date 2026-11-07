#!/usr/bin/env python3
"""Run the upstream RLNN CLI with local provenance and metrics."""
import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'v000_rlnn_baseline'


def parse_training(text):
    records = []
    for line in text.splitlines():
        pairs = dict(re.findall(r'(Epoch|Iteration|loss|valid_obj|valid_time):\s*(\S+)', line))
        if pairs.get('valid_obj') in (None, 'None'):
            continue
        obj = float(pairs['valid_obj'])
        records.append({'epoch': int(pairs['Epoch']), 'step': int(pairs['Iteration']) if int(pairs.get('Iteration',0)) > 0 else None, 'train/loss': float(pairs['loss']),
                        'val/raw_obj': obj, 'val/independent_set_size': -obj,
                        'val/time': float(pairs['valid_time'])})
    return records


def graph_count(path):
    folder = Path(path)
    manifest = folder / 'tensor_manifest.json'
    return len(json.loads(manifest.read_text())['graphs']) if manifest.exists() else len(list(folder.glob('*gpickle')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'eval', 'smoke'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--version', default=VERSION)
    parser.add_argument('--dataset', help='dataset label for run provenance')
    parser.add_argument('--kind', choices=['single_gpu','smoke'], help='result classification; inferred from mode by default')
    parser.add_argument('--checkpoint', type=Path, help='directory containing best.pt (evaluation)')
    control, forwarded = parser.parse_known_args()
    for name in ('run', 'version'):
        value = getattr(control, name)
        if Path(value).name != value or value in ('', '.', '..'):
            parser.error(f'--{name} must be a single directory name')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
    sys.path.insert(0, str(ROOT / 'models'))
    upstream = importlib.import_module('main')
    upstream_parser = upstream.build_parser()
    upstream_parser.set_defaults(method='RLNN', loss='erdoes', num_l=5, num_h=128,
        problem='mis', epochs=50, batch_size=1024, graph_batch_size=13, lr=0.0001, num_t=100, num_k=20,
        num_tp=50, num_kp=10, num_d=5, beta=1.02, lambd=0.5,
        mixed_precision='no', num_workers=8,
        train_path=str(ROOT / 'models/data/MIS_tensors/RB_small_train'),
        valid_path=str(ROOT / 'models/data/MIS_tensors/RB_small_val'),
        test_path=str(ROOT / 'models/data/MIS_tensors/RB_small_test'),
        train_samples=1000, valid_samples=100, test_samples=-1)
    args = upstream_parser.parse_args(forwarded)
    if args.method != 'RLNN' or args.problem != 'mis' or args.skip_decode:
        parser.error('project baseline requires RLNN, MIS and enabled feasibility decoding')
    if control.mode == 'eval' and control.checkpoint is None:
        parser.error('evaluation requires --checkpoint DIR containing best.pt')
    run = ROOT / 'outputs' / control.version / control.run
    run.mkdir(parents=True, exist_ok=False)
    if control.mode == 'smoke':
        import networkx as nx
        for split, count in [('train', 2), ('val', 1), ('test', 1)]:
            folder = run / 'data' / split
            folder.mkdir(parents=True)
            for i in range(count):
                graph = nx.gnp_random_graph(8, 0.3, seed=100 * ['train','val','test'].index(split)+i)
                with (folder / f'{i}.gpickle').open('wb') as stream:
                    pickle.dump(graph, stream)
            setattr(args, {'train':'train_path','val':'valid_path','test':'test_path'}[split], str(folder))
        args.epochs, args.batch_size, args.num_k, args.num_tp, args.num_t = 1, 2, 2, 2, 2
        args.train_samples, args.valid_samples, args.test_samples = 2, 1, 1
        args.graph_batch_size = min(args.graph_batch_size, 2)
        args.num_workers = 0
    for name in ['train_path', 'valid_path', 'test_path']:
        path = Path(getattr(args, name))
        path = path if path.is_absolute() else ROOT / path
        setattr(args, name, str(path.resolve()))
    args.save_dir = str(control.checkpoint.resolve() if control.checkpoint else run / 'checkpoints')
    args.log_file = str(run / 'evaluation.txt')
    args.do_train = control.mode != 'eval'
    import torch
    kind = control.kind or ('smoke' if control.mode == 'smoke' else 'single_gpu')
    metadata = dict(version=control.version, run=control.run, task='MIS', dataset=control.dataset or ('tiny-ER8' if control.mode == 'smoke' else 'RB200-300'),
        kind=kind, mode=control.mode,
        training_split=args.train_path, validation_split=args.valid_path, test_split=args.test_path,
        config=vars(args).copy(), seed=args.seed, started_at=datetime.now(timezone.utc).isoformat(),
        device={'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'cpu_threads': torch.get_num_threads(), 'allocator_config': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', ''), 'allocator_backend': torch.cuda.memory.get_allocator_backend() if torch.cuda.is_available() else 'cpu', 'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'},
        dataset_counts={name:graph_count(getattr(args,name)) for name in ['train_path','valid_path','test_path']},
        reference=None, status='running')
    metadata_path = run / 'run.json'
    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    save_metadata()
    records = []
    resume_epoch = 0
    if args.resume:
        resume_state = torch.load(args.resume, map_location='cpu', weights_only=False)
        resume_epoch = resume_state['epoch']
        lines = [' '.join(f'{key}: {value}' for key,value in row.items()) for row in resume_state['history']]
        records = parse_training('\n'.join(lines))
        source_meta_path = Path(resume_state.get('source_run', '')) / 'run.json'
        source_meta = json.loads(source_meta_path.read_text()) if source_meta_path.is_file() else {}
        metadata['resume'] = {'source': args.resume, 'epoch': resume_epoch, 'mode': resume_state['resume_mode'], 'optimizer_reset': resume_state['optimizer'] is None, 'random_state_reset': resume_state['rng'] is None, 'inherited_config': resume_state.get('config'), 'inherited_device': resume_state.get('device') or source_meta.get('device'), 'source_run': resume_state.get('source_run')}
        save_metadata()
    started = time.perf_counter()
    try:
        modes = ['train', 'eval'] if control.mode == 'smoke' else [control.mode]
        for mode in modes:
            args.do_train = mode == 'train'
            required = ['train_path', 'valid_path'] if args.do_train else ['test_path']
            for name in required:
                if not graph_count(getattr(args,name)):
                    raise ValueError(f'{name}: no graphs in {getattr(args,name)}')
            if not args.do_train and not (Path(args.save_dir) / 'best.pt').is_file():
                raise FileNotFoundError(Path(args.save_dir) / 'best.pt')
            command = [sys.executable, '-u', 'main.py']
            for key, value in vars(args).items():
                if isinstance(value, bool):
                    if value: command.append('--'+key)
                elif value is not None:
                    command.extend(['--'+key, str(value)])
            metadata.setdefault('commands', []).append(command)
            save_metadata()
            with (run / f'{mode}.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=ROOT / 'models', stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                lines = []
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end='', flush=True); lines.append(line)
                code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
            if mode == 'train':
                parsed = parse_training(''.join(lines))
                if not parsed: raise ValueError('training log contains no completed validation')
                records.extend(row for row in parsed if row['epoch'] > resume_epoch)
            else:
                text = Path(args.log_file).read_text()
                obj = float(re.search(r'Avg\. obj:\s*(\S+)', text)[1])
                seconds = float(re.search(r'Avg\. time:\s*(\S+)', text)[1])
                records.append({'test/raw_obj':obj, 'test/independent_set_size':-obj, 'test/time':seconds})
        records.append({'run/wall_time': time.perf_counter()-started})
        folder = run / 'metrics'; folder.mkdir()
        columns = ['epoch'] + sorted({key for record in records for key in record if key != 'epoch'})
        with (folder / 'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(records)
        metadata['status'] = 'completed'
    except Exception as error:
        metadata.update(status='failed', error=str(error))
        raise
    finally:
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), wall_time_s=time.perf_counter()-started)
        save_metadata()


if __name__ == '__main__':
    main()
