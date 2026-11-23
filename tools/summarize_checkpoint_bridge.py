"""Pair successive solver, particle-count, and checkpoint comparisons on RB-small."""
import argparse
import json
from pathlib import Path

from compare_mis_runs import compare, read_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    author_checkpoint = 'models/data/author_rld4co/checkpoints/erdoes/rlnn_rb_small_mis'
    local = Path('outputs/v11/fused_pave_02s_deadline')
    released = Path('outputs/v11/reference_deadline_20261119_followup/author_rlnn/per_graph.csv')
    seeds = (1017, 1019, 1021)
    expected = {(f'idx_{i}', str(seed), '0') for seed in seeds for i in range(1000)}
    for k in (20, 200):
        arm = args.run_dir / f'author_k{k}'
        rows = read_rows(arm)
        if rows.keys() != expected:
            raise ValueError(f'Unmatched author graph--seed pairs in {arm}')
        if any((r[2] is None and r[0] != 0) or (r[2] is not None and r[2] > 200) for r in rows.values()):
            raise ValueError(f'Scored late or missing set in {arm}')
        for seed in seeds:
            path = arm / f'batch_{seed}/seed_{seed}/padded/padded_ref0.json'
            config = json.loads(path.read_text(encoding='utf-8'))
            required = dict(checkpoint_dir=author_checkpoint, num_k=k, decode_top=4, dtype='fp16',
                            dense=1, compile_model=True, fixed_steps=0, seeds=[seed],
                            decode_schedule='midpoint', pave=False, budgets=[0.2])
            if any(config[key] != value for key, value in required.items()):
                raise ValueError(f'Unexpected bridge configuration in {path}')
    k20 = args.run_dir / 'author_k20'
    k200 = args.run_dir / 'author_k200'
    results = {
        'fused_protocol_vs_released_at_author_k20': compare(k20, released),
        'author_k200_vs_author_k20_under_fused_protocol': compare(k200, k20),
        'local_vs_author_checkpoint_at_fused_k200': compare(local, k200, candidate_glob='pave_off_*/v11_rows.csv'),
        'complete_configuration_vs_released': compare(local, released, candidate_glob='pave_off_*/v11_rows.csv'),
    }
    (args.run_dir / 'summary.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
