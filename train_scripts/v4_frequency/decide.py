"""Apply the v4 validation gate before expanding the training budget."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/v4_frequency'


def main():
    values = {}
    for path in OUT.glob('*/*/run.json'):
        if path.parent.parent.name.startswith('smoke'):
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('mode') != 'eval' or data.get('status') != 'completed' or data.get('split') != 'validation':
            continue
        if any(data.get('interventions', {}).values()):
            continue
        config = data['config']
        key = (config['mixed_precision'], data['protocol'], data['post'], data['method'], data['train_seed'], data.get('extended_search', False))
        values[key] = data['mean_size']
    result = dict(stage='A1', status='pending', primary='FP32 fixed-work without PAVE',
                  efficiency='FP16 0.5-second budget with frozen PAVE', comparisons=[])
    tests = [('no', 'fixed_work', 'none', control, False) for control in ('v4a_mlp', 'v4a_message')]
    tests += [('fp16', 'fixed_time', 'pave', control, False) for control in ('baseline', 'v4a_mlp', 'v4a_message')]
    tests += [('fp16', 'fixed_time', 'pave', 'baseline', True)]
    missing = []
    for precision, protocol, post, control, extended in tests:
        keys = [(precision, protocol, post, label, seed, is_extended)
                for label, is_extended in (('v4a', False), (control, extended)) for seed in range(3)]
        missing.extend(key for key in keys if key not in values)
        if any(key not in values for key in keys):
            continue
        deltas = [round(values[precision, protocol, post, 'v4a', seed, False] - values[precision, protocol, post, control, seed, extended], 8) for seed in range(3)]
        positive = sum(delta > 0 for delta in deltas)
        passed = mean(deltas) > 0 and (positive >= 2 if protocol == 'fixed_work' else True)
        result['comparisons'].append(dict(precision=precision, protocol=protocol, post=post,
                                          control=control, control_extended_search=extended, deltas=deltas, mean_delta=mean(deltas),
                                          positive_seeds=positive, passed=passed))
    if missing:
        result['missing'] = sorted(set(missing))
    else:
        passed = all(item['passed'] for item in result['comparisons'])
        result.update(status='passed' if passed else 'stop_current_adapter',
                      next_stage='A2_50_epochs' if passed else 'mechanism_analysis',
                      pave_change=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'decision.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
