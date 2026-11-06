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
        records.append({'epoch': int(pairs['Epoch']), 'train/loss': float(pairs['loss']),
                        'val/raw_obj': obj, 'val/independent_set_size': -obj,
                        'val/time': float(pairs['valid_time'])})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['train', 'eval', 'smoke'])
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoint', type=Path, help='directory containing best.pt (evaluation)')
    control, forwarded = parser.parse_known_args()
    if Path(control.run).name != control.run or control.run in ('.', '..'):
        parser.error('--run must be a single directory name')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
    sys.path.insert(0, str(ROOT / 'models'))
    upstream = importlib.import_module('main')
    upstream_parser = upstream.build_parser()
    upstream_parser.set_defaults(method='RLNN', loss='erdoes', num_l=5, num_h=128,
        problem='mis', epochs=50, batch_size=32, lr=0.0001, num_t=100, num_k=20,
        num_tp=50, num_kp=10, num_d=5, beta=1.02, lambd=0.5,
        mixed_precision='no', num_workers=0,
        train_path=str(ROOT / 'models/data/MIS/RB_small_train'),
        valid_path=str(ROOT / 'models/data/MIS/RB_small_val'),
        test_path=str(ROOT / 'models/data/MIS/RB_small_test'),
        train_samples=1000, valid_samples=100, test_samples=-1)
    args = upstream_parser.parse_args(forwarded)
    if args.method != 'RLNN' or args.problem != 'mis' or args.skip_decode:
        parser.error('project baseline requires RLNN, MIS and enabled feasibility decoding')
    if control.mode == 'eval' and control.checkpoint is None:
        parser.error('evaluation requires --checkpoint DIR containing best.pt')
    run = ROOT / 'outputs' / VERSION / control.run
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
    for name in ['train_path', 'valid_path', 'test_path']:
        path = Path(getattr(args, name))
        path = path if path.is_absolute() else ROOT / path
        setattr(args, name, str(path.resolve()))
    args.save_dir = str(control.checkpoint.resolve() if control.checkpoint else run / 'checkpoints')
    args.log_file = str(run / 'evaluation.txt')
    args.do_train = control.mode != 'eval'
    import torch
    metadata = dict(version=VERSION, run=control.run, task='MIS', dataset='RB200-300' if control.mode != 'smoke' else 'tiny-ER8',
        kind='smoke' if control.mode == 'smoke' else 'single_gpu', mode=control.mode,
        training_split=args.train_path, validation_split=args.valid_path, test_split=args.test_path,
        config=vars(args).copy(), seed=args.seed, started_at=datetime.now(timezone.utc).isoformat(),
        device={'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'], 'torch': torch.__version__,
                'cuda': torch.version.cuda, 'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'},
        dataset_counts={name:len(list(Path(getattr(args,name)).glob('*gpickle'))) for name in ['train_path','valid_path','test_path']},
        reference=None, status='running')
    metadata_path = run / 'run.json'
    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    save_metadata()
    records = []
    started = time.perf_counter()
    try:
        modes = ['train', 'eval'] if control.mode == 'smoke' else [control.mode]
        for mode in modes:
            args.do_train = mode == 'train'
            required = ['train_path', 'valid_path'] if args.do_train else ['test_path']
            for name in required:
                if not list(Path(getattr(args,name)).glob('*gpickle')):
                    raise ValueError(f'{name}: no .gpickle graphs in {getattr(args,name)}')
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
                records.extend(parsed)
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
