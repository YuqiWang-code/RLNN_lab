"""Mix rule-visited and PAVE-visited states at a fixed training-set size."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import torch


def load_labels(path):
    path = path / 'states.pt' if path.is_dir() else path
    metadata = json.loads((path.parent / 'run.json').read_text(encoding='utf-8'))
    states = torch.load(path, map_location='cpu', weights_only=False)
    if metadata['status'] != 'completed' or metadata['mode'] != 'labels' or metadata['split'] != 'train':
        raise ValueError(f'{path} must contain completed training labels')
    if not states or any(state['split'] != 'train' for state in states):
        raise ValueError(f'{path} must contain nonempty training states only')
    return path.resolve(), states, metadata


def mix_states(original, policy, seed):
    rng = random.Random(seed)
    sources = {'original': defaultdict(list), 'policy': defaultdict(list)}
    for source, states in (('original', original), ('policy', policy)):
        for state in states:
            sources[source][state['origin']].append(state)
    mixed = []
    source_origins = {'original': {}, 'policy': {}}
    for origin in sorted(sources['original']):
        base = sources['original'][origin]
        visited = sources['policy'][origin]
        policy_count = min(len(base) // 2, len(visited))
        original_count = len(base) - policy_count
        for source, states, count in (('original', base, original_count), ('policy', visited, policy_count)):
            chosen = rng.sample(states, count)
            mixed.extend(dict(state, mix_source=source) for state in chosen)
            source_origins[source][origin] = count
    if not sum(source_origins['policy'].values()):
        raise ValueError('No PAVE-visited states can be included at the original origin counts')
    rng.shuffle(mixed)
    return mixed, source_origins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    original_path, original, original_run = load_labels(args.original)
    policy_path, policy, policy_run = load_labels(args.policy)
    for key in ('horizon', 'anchored', 'candidate_cap'):
        if original_run['config'][key] != policy_run['config'][key]:
            raise ValueError(f'Label continuation settings differ: {key}')
    if policy_run['config']['policy'] != 'pave':
        raise ValueError('The policy source must be collected with --policy pave')
    mixed, source_origins = mix_states(original, policy, args.seed)
    informative = [state for state in mixed if len(state['values']) and
                   float(state['values'].max() - state['values'].min()) > 0]
    config = dict(original_run['config'], policy='mixture', mix_seed=args.seed,
                  original_data=str(original_path), policy_data=str(policy_path),
                  value_checkpoint=policy_run['config']['value_checkpoint'])
    samples = {(state['graph_id'], state['backbone_seed'], state['stream'], state['search_seed'])
               for state in mixed}
    metadata = dict(version='v3', status='completed', mode='labels', split='train',
                    created_at=datetime.now(timezone.utc).isoformat(), config=config,
                    samples=len(samples), state_count=len(mixed), graph_count=len({s['graph_id'] for s in mixed}),
                    informative_states=len(informative),
                    informative_graphs=len({state['graph_id'] for state in informative}),
                    origin_counts=dict(Counter(state['origin'] for state in mixed)),
                    source_counts={source: sum(counts.values()) for source, counts in source_origins.items()},
                    source_origin_counts=source_origins,
                    available_source_states=dict(original=len(original), policy=len(policy)),
                    sampling='origin-stratified without replacement within each source')
    args.result_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mixed, args.result_dir / 'states.pt')
    text = json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + '\n'
    (args.result_dir / 'run.json').write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
