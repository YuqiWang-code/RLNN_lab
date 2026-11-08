"""Cache frozen RLNN decoded starts for CPU action census and training labels."""
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
from time import perf_counter

import torch

from eval_v2 import TRAIN, VALID, TEST_CACHE, dataset_at, load_policy, policy_parser, search, sync, write_run
from mis_v2_exchange import ExchangeGraph, nodes


@torch.no_grad()
def main():
    parser = policy_parser()
    parser.add_argument('--data_split', choices=('train', 'validation', 'test'), default='validation')
    parser.add_argument('--stream', type=int, default=0)
    args = parser.parse_args()
    args.split = 'test' if args.data_split == 'test' else 'validation'
    if args.test_path is None:
        args.test_path = str({'train': TRAIN, 'validation': VALID, 'test': TEST_CACHE}[args.data_split])
    solver = load_policy(args)
    device = solver.accelerator.device
    dataset = dataset_at(args.test_path, solver.dtype, args.test_samples, official=args.data_split == 'test')
    args.result_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    records, small = [], []
    for index, graph in enumerate(dataset):
        edges = graph.edge_index.clone()
        topology = ExchangeGraph(graph.num_nodes, edges)
        graph_id = f'{args.data_split}/' + (f'idx_{index}' if args.data_split == 'test' else Path(dataset.files[index]).stem)
        seed = args.seed + args.stream * 100000 + index
        sync(device)
        tick = perf_counter()
        selected, steps = search(solver, graph.to(device), topology, seed)
        sync(device)
        seconds = perf_counter() - tick
        record = dict(graph_id=graph_id, split=args.data_split, graph_index=index,
                      backbone_seed=args.train_seed, search_seed=seed, stream=args.stream,
                      num_nodes=graph.num_nodes, edge_index=edges, selected=selected,
                      search_steps=steps, search_time_s=seconds)
        records.append(record)
        small.append({key: value for key, value in record.items() if key not in ('edge_index', 'selected')})
        small[-1].update(base_size=selected.bit_count(), selected_nodes=' '.join(map(str, nodes(selected))))
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(f'{args.data_split} seed={args.train_seed} stream={args.stream} {index+1}/{len(dataset)} '
                  f'mean={sum(r["base_size"] for r in small)/len(small):.4f}', flush=True)
    torch.save(records, args.result_dir / 'starts.pt')
    with (args.result_dir / 'per_graph.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(small[0]))
        writer.writeheader()
        writer.writerows(small)
    write_run(args.result_dir, dict(version='v3', status='completed', mode='starts',
              split=args.data_split, backbone_seed=args.train_seed, stream=args.stream,
              samples=len(records), config=vars(args), started_at=started_at,
              finished_at=datetime.now(timezone.utc).isoformat(), wall_time_s=perf_counter()-started,
              mean_size=sum(r['base_size'] for r in small)/len(small), gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))


if __name__ == '__main__':
    main()
