"""Compare frozen-start platform quality before admitting graph encoders."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import pickle
from statistics import mean
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'models'))
import torch
from mis_v3_search import SearchGraph, platform_search
from mis_v3_value import load_pave_v3


def worker(payload):
    record, output, backbone = pickle.loads(payload)
    torch.set_num_threads(1)
    graph = SearchGraph(record['num_nodes'], record['edge_index'], record['search_seed'])
    rows = []
    choices = [('none', 2, None), ('rule23', 2, None)]
    choices += [(rule, horizon, None) for horizon in (1,2) for rule in ('random','degree','coverage')]
    for horizon in (1,2):
        checkpoint = output/f'stats_h{horizon}_seed{record["backbone_seed"]}/checkpoints/best.pt'
        if checkpoint.is_file():
            choices.append(('stats', horizon, load_pave_v3(checkpoint)))
    for mode in ('message','frequency','stats_adapted'):
        checkpoint = output/f'{mode}_h1_seed{record["backbone_seed"]}/checkpoints/best.pt'
        if checkpoint.is_file() and (checkpoint.parent.parent/'run.json').is_file():
            choices.append((mode, 1, load_pave_v3(checkpoint)))
    for policy, horizon, model in choices:
        tick = perf_counter()
        selected = record['selected']
        result = None if policy=='none' else platform_search(graph, selected,
                    choose=model.choose if model else policy, horizon=horizon,
                    outer_rounds=0 if policy=='rule23' else 8)
        final = result.best if result else selected
        if not graph.feasible(final):
            raise ValueError('Invalid cached-validation result')
        rows.append(dict(graph_id=record['graph_id'], backbone=backbone,
            backbone_seed=record['backbone_seed'], policy=policy, horizon=horizon,
            base_size=selected.bit_count(), rule_size=result.rule_size if result else selected.bit_count(),
            final_size=final.bit_count(), post_time_s=perf_counter()-tick,
            neutral_decisions=result.neutral_decisions if result else 0))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/v3'))
    parser.add_argument('--workers',type=int,default=24)
    args=parser.parse_args()
    torch.set_num_threads(1)
    records=[]
    for backbone in ('original','tuned'):
        for seed in range(3):
            name=f'starts_validation_seed{seed}_stream0' if backbone=='original' else f'starts_tuned_seed{seed}'
            records += [pickle.dumps((r,args.output,backbone),protocol=5)
                        for r in torch.load(args.output/name/'starts.pt',weights_only=False)]
    tick=perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=[row for group in pool.map(worker,records) for row in group]
    result=args.output/'validation_platform'
    result.mkdir(exist_ok=True)
    with (result/'per_graph.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={}
    for backbone in ('original','tuned'):
        variants=sorted({(r['policy'],r['horizon']) for r in rows if r['backbone']==backbone})
        summary[backbone]={}
        for policy,horizon in variants:
            group=[r for r in rows if r['backbone']==backbone and r['policy']==policy and r['horizon']==horizon]
            summary[backbone][f'{policy}_h{horizon}']=dict(mean_size=mean(r['final_size'] for r in group),
                by_seed={str(seed):mean(r['final_size'] for r in group if r['backbone_seed']==seed) for seed in range(3)})
    backbone=max(summary,key=lambda b:max(v['mean_size'] for k,v in summary[b].items() if k.startswith(('random','degree','coverage'))))
    rules={key:value for key,value in summary[backbone].items() if key.startswith(('random','degree','coverage'))}
    strongest=max(rules,key=lambda key:rules[key]['mean_size'])
    stats={key:value for key,value in summary[backbone].items() if key in ('stats_h1','stats_h2')}
    selected_stats=max(stats,key=lambda key:stats[key]['mean_size']) if stats else None
    delta=stats[selected_stats]['mean_size']-rules[strongest]['mean_size'] if selected_stats else None
    choice=dict(backbone=backbone,strongest_rule=strongest,stats=selected_stats,
                delta_vs_strongest_rule=delta,train_encoders=delta is not None and delta>0)
    learned={key:value for key,value in summary[backbone].items()
             if key.startswith(('stats','message','frequency'))}
    choice['best_learned']=max(learned,key=lambda key:learned[key]['mean_size']) if learned else None
    if 'message_h1' in learned and 'frequency_h1' in learned:
        differences=[learned['frequency_h1']['by_seed'][str(seed)]-learned['message_h1']['by_seed'][str(seed)]
                     for seed in range(3)]
        choice['frequency_minus_message_by_seed']=differences
        choice['train_exchange_band']=all(delta>0 for delta in differences)
    metadata=dict(version='v3',status='completed',mode='validation_platform',split='validation',
                  samples=600,wall_time_s=perf_counter()-tick,summaries=summary,choice=choice,
                  timing='Parallel cached-start quality selection; post timing is diagnostic only')
    (result/'run.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (args.output/'model_decision.json').write_text(json.dumps(choice,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
