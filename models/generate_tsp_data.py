"""Generate uniform Euclidean TSP instances with LKH reference tours."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np


def solve_instance(item):
    index, seed, nodes, trials, executable, scratch = item
    points = np.random.default_rng(np.random.SeedSequence([seed, index])).random((nodes, 2))
    coordinates = np.rint(points * 1_000_000).astype(np.int64)
    with tempfile.TemporaryDirectory(prefix='tsp_', dir=scratch) as directory:
        directory = Path(directory)
        problem = directory / 'instance.tsp'
        tour_file = directory / 'solution.tour'
        problem.write_text('\n'.join([
            'NAME: uniform', 'TYPE: TSP', f'DIMENSION: {nodes}', 'EDGE_WEIGHT_TYPE: EUC_2D',
            'NODE_COORD_SECTION',
            *[f'{i + 1} {x} {y}' for i, (x, y) in enumerate(coordinates)], 'EOF', '']), encoding='ascii')
        parameters = directory / 'instance.par'
        parameters.write_text(f'PROBLEM_FILE = {problem}\nOUTPUT_TOUR_FILE = {tour_file}\nRUNS = 1\nMAX_TRIALS = {trials}\nSEED = {seed + index}\nTRACE_LEVEL = 0\n', encoding='ascii')
        subprocess.run([executable, str(parameters)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tokens = tour_file.read_text().split('TOUR_SECTION', 1)[1].split()
        tour = []
        for token in tokens:
            if token in {'-1', 'EOF'}:
                break
            tour.append(int(token))
        if sorted(tour) != list(range(1, nodes + 1)):
            raise ValueError(f'LKH returned an invalid tour for instance {index}')
    tour.append(tour[0])
    line = ' '.join(f'{value:.12f}' for value in points.reshape(-1)) + ' output ' + ' '.join(map(str, tour))
    route = points[np.array(tour) - 1]
    cost = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    return line, cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--nodes', type=int, default=100)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--lkh', default='/usr/local/bin/LKH')
    parser.add_argument('--scratch', type=Path, default=Path('../tmp'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tasks = ((i, args.seed, args.nodes, args.trials, args.lkh, str(args.scratch.resolve())) for i in range(args.samples))
    costs = []
    with args.output.open('x', encoding='utf-8') as output, ProcessPoolExecutor(args.workers) as executor:
        for i, (line, cost) in enumerate(executor.map(solve_instance, tasks, chunksize=8), 1):
            output.write(line + '\n')
            costs.append(cost)
            if i % 500 == 0 or i == args.samples:
                print(f'{i}/{args.samples} instances, {time.time() - started:.1f}s', flush=True)
    manifest = {**vars(args), 'output': str(args.output), 'scratch': str(args.scratch),
                'distribution': 'independent U([0,1]^2)', 'coordinate_scale': 1_000_000,
                'solver_runs': 1, 'mean_reference_cost': float(np.mean(costs)),
                'generation_seconds': time.time() - started, 'numpy_version': np.__version__}
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
