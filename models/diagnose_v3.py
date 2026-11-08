"""Parallel action census and statewise labels for frozen-policy MIS starts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
from functools import lru_cache
import json
import pickle
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from mis_v2_exchange import action_values as improving_values
from mis_v3_search import (SearchGraph, candidate_actions, improving_actions, choose_action,
                           repair, action_values, platform_search)
from mis_v3_value import build_state, load_pave_v3

RULES = ('coverage', 'degree', 'random')


@lru_cache(maxsize=1)
def value_policy(checkpoint):
    return load_pave_v3(checkpoint, device='cpu')


def indices(graph, selected, actions):
    return {rule: actions.index(choose_action(graph, selected, actions, rule)) if actions else -1
            for rule in RULES}


def describe(record, variant, selected, actions, values, full_count, results=None, origin='initial'):
    counts = Counter(values)
    ties = sum(count*(count-1)//2 for count in counts.values())
    graph = record['_graph']
    rules = indices(graph, selected, actions)
    row = dict(graph_id=record['graph_id'], backbone_seed=record['backbone_seed'], stream=record['stream'],
               search_seed=record['search_seed'], origin=origin, variant=variant,
               base_size=record['selected'].bit_count(), state_size=selected.bit_count(),
               full_candidates=full_count, candidates=len(actions), pairs=len(actions)*(len(actions)-1)//2,
               non_tied_pairs=len(actions)*(len(actions)-1)//2-ties,
               informative=int(bool(values) and min(values)<max(values)),
               oracle_gain=max(values, default=0), mean_gain=mean(values) if values else 0)
    for rule, index in rules.items():
        row[rule+'_gain'] = values[index] if index >= 0 else 0
    if results is not None:
        row.update(returned_to_start=mean(r.returned_to_start for r in results) if results else 0,
                   ended_below_start=mean(r.ended_below_start for r in results) if results else 0,
                   capped_rollouts=sum(r.capped for r in results))
    return row


def state_label(record, selected, actions, remaining, values, origin):
    state = build_state(record['_graph'], selected, actions, remaining, include_nodes=True)
    state.update(graph_id=record['graph_id'], split=record['split'], origin=origin,
                 backbone_seed=record['backbone_seed'], stream=record['stream'],
                 search_seed=record['search_seed'], selected=selected,
                 values=torch.tensor(values, dtype=torch.float32),
                 rule_indices=indices(record['_graph'], selected, actions),
                 heuristic_index=-1)
    state['heuristic_index'] = state['rule_indices']['coverage']
    return state


def worker(payload):
    record, options = pickle.loads(payload)
    torch.set_num_threads(1)
    tick = perf_counter()
    graph = SearchGraph(record['num_nodes'], record['edge_index'], seed=record['search_seed'])
    record['_graph'] = graph
    rows, states, action_rows = [], [], []
    if options['mode'] == 'census':
        for variant, union in (('D0_fallback', False), ('D1_union', True)):
            actions = improving_actions(graph, record['selected'], cap=None, union=union)
            values = [value-record['selected'].bit_count() for value in improving_values(
                graph, record['selected'], actions, cap=48, rounds=3, larger=True)]
            rows.append(describe(record, variant, record['selected'], actions, values, len(actions)))
        initial = repair(graph, record['selected'], rounds=3, extend_first=False)
        selected = initial.work
        supplements = 0
        while improving_actions(graph, selected, cap=1) and supplements < 8:
            selected = repair(graph, selected).work
            supplements += 1
        if improving_actions(graph, selected, cap=1):
            raise RuntimeError(f'{record["graph_id"]}: census did not reach a strict stopping state')
        full = candidate_actions(graph, selected, cap=None, anchored=options['anchored'])
        pool = candidate_actions(graph, selected, cap=options['candidate_cap'], anchored=options['anchored'])
        kept = [full.actions.index(action) for action in pool.actions]
        horizons = (1, 2) if not options['anchored'] else (2,)
        for horizon in horizons:
            variant = f'D{horizon+1}_neutral_h{horizon}' if not options['anchored'] else 'D4_anchored_h2'
            full_values, full_results = action_values(graph, selected, full.actions, horizon=horizon,
                                                      return_rollouts=True, candidate_cap=options['candidate_cap'])
            values = [full_values[index] for index in kept]
            results = [full_results[index] for index in kept]
            row = describe(record, variant, selected, pool.actions, values, full.total_count, results)
            row.update(full_oracle_gain=max(full_values, default=0),
                       truncation_oracle_loss=max(full_values, default=0)-max(values, default=0),
                       rule_supplements=supplements)
            rows.append(row)
            if horizon == options['horizon']:
                states.append(state_label(record, selected, pool.actions, 1.0, values, 'initial'))
            for index, (action, value, result) in enumerate(zip(full.actions, full_values, full_results)):
                action_rows.append(dict(graph_id=record['graph_id'], backbone_seed=record['backbone_seed'],
                    stream=record['stream'], variant=variant, candidate=index, kept=int(index in kept),
                    removed=' '.join(map(str, action.removed)), added=' '.join(map(str, action.added)),
                    gain=value, final_size=result.best.bit_count(), exchanges=result.exchanges,
                    returned_to_start=result.returned_to_start, ended_below_start=result.ended_below_start,
                    rolled_back=result.rolled_back, segments=result.segments))
    else:
        def observe(graph, selected, pool, history, outer, remaining):
            if len(states) >= options['states_per_start']:
                return
            values, results = action_values(graph, selected, pool.actions, horizon=options['horizon'],
                                             return_rollouts=True, candidate_cap=options['candidate_cap'])
            origin = 'initial' if not states else 'intermediate'
            label = state_label(record, selected, pool.actions, remaining, values, origin)
            label['outer_step'] = outer
            states.append(label)
            rows.append(describe(record, f'labels_h{options["horizon"]}', selected, pool.actions,
                                 values, pool.total_count, results, origin))
        policy = (value_policy(options['value_checkpoint']).choose
                  if options['policy'] == 'pave' else options['policy'])
        platform_search(graph, record['selected'], choose=policy, horizon=options['horizon'],
                        candidate_cap=options['candidate_cap'], anchored=options['anchored'], on_state=observe)
    return pickle.dumps((rows, states, action_rows, perf_counter()-tick), protocol=5)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summaries = {}
    for variant in sorted({row['variant'] for row in rows}):
        group = [row for row in rows if row['variant'] == variant]
        gains = {rule: mean(row[rule+'_gain'] for row in group) for rule in RULES}
        strongest = max(gains, key=gains.get)
        oracle = mean(row['oracle_gain'] for row in group)
        summaries[variant] = dict(states=len(group), graphs=len({row['graph_id'] for row in group}),
            available=sum(row['candidates']>0 for row in group),
            informative=sum(row['informative'] for row in group),
            informative_graphs=len({row['graph_id'] for row in group if row['informative']}),
            non_tied_pairs=sum(row['non_tied_pairs'] for row in group),
            positive_states=sum(row['oracle_gain']>0 for row in group),
            positive_coverage=mean(row['oracle_gain']>0 for row in group),
            mean_candidates=mean(row['candidates'] for row in group),
            mean_oracle_gain=oracle, rule_gains=gains, strongest_rule=strongest,
            oracle_gap=oracle-gains[strongest],
            mean_truncation_loss=mean(row.get('truncation_oracle_loss',0) for row in group))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--starts', nargs='+', type=Path, required=True)
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('census','collect'), default='census')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--horizon', type=int, choices=(1,2), default=2)
    parser.add_argument('--candidate-cap', type=int, default=32)
    parser.add_argument('--states-per-start', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--anchored', action='store_true')
    parser.add_argument('--policy', choices=(*RULES, 'pave'), default='coverage')
    parser.add_argument('--value-checkpoint', type=Path)
    args = parser.parse_args()
    if args.policy == 'pave' and args.value_checkpoint is None:
        parser.error('--policy pave requires --value-checkpoint')
    torch.set_num_threads(1)
    records = [record for path in args.starts for record in torch.load(path, weights_only=False)]
    if args.limit:
        records = records[:args.limit]
    args.result_dir.mkdir(parents=True, exist_ok=True)
    tick = perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    options = vars(args).copy()
    rows, states, action_rows, durations = [], [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = (pickle.dumps((record, options), protocol=5) for record in records)
        for index, result in enumerate(pool.map(worker, tasks)):
            rr, ss, aa, duration = pickle.loads(result)
            rows.extend(rr); states.extend(ss); action_rows.extend(aa); durations.append(duration)
            if (index+1)%25 == 0 or index+1 == len(records):
                print(f'{args.mode} {index+1}/{len(records)} starts, {len(states)} states, '
                      f'{sum(bool(len(s["values"])) and float(s["values"].max()-s["values"].min())>0 for s in states)} informative', flush=True)
    torch.save(states, args.result_dir/'states.pt')
    write_csv(args.result_dir/'per_state.csv', rows)
    write_csv(args.result_dir/'per_action.csv', action_rows)
    metadata = dict(version='v3',status='completed',mode='opportunities' if args.mode=='census' else 'labels',
        split=records[0]['split'], samples=len(records), state_count=len(states),config=vars(args),
        started_at=started, finished_at=datetime.now(timezone.utc).isoformat(), wall_time_s=perf_counter()-tick,
        mean_cpu_task_s=mean(durations),summaries=summarize(rows),
        informative_states=sum(bool(len(s['values'])) and float(s['values'].max()-s['values'].min())>0 for s in states),
        informative_graphs=len({s['graph_id'] for s in states if len(s['values']) and float(s['values'].max()-s['values'].min())>0}))
    (args.result_dir/'run.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps(metadata['summaries'],ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
