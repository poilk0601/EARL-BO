"""
Run EARL-BO on a discrete HPO-B search space, several seeds in parallel (one process per seed).

Defaults = the paper's setting for HPO-B search space 7200 (19 hyperparameters, 811 configurations; data/hpob_7200.csv):
50 initial points, 50 decisions, horizon 3, seeds 0-9. Initial designs: np.random.seed(0), then ten draws of n0 distinct
grid indices; seed s uses draw s, so the seeds are 0-9.

Outputs (in --out):
  regret_seed{s}.csv   iteration, regret, decision_time. Row 0 is the initial design (no decision time); regret = grid
                       maximum - incumbent, with y normalised over the grid so the maximum is 1; decision_time in seconds.
  summary.csv          per iteration: mean and std (over the finished seeds) of regret and decision_time, n_seeds
  logs/seed{s}.log     stdout of seed s (training progress, the grid point chosen)
  config.json          the arguments of the run and the ten initial index sets
"""
import os

# single-threaded BLAS in every worker (set before numpy / torch are imported)
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hpob_grid import Grid, initial_sets  # noqa: E402

N_DESIGNS = 10


def parse_seeds(text):
    """'0-9', '0,3,7', or both forms combined."""
    seeds = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-')
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _worker(job):
    seed, idx0, cfg = job
    import torch
    torch.set_num_threads(1)
    import earlbo_discrete as m
    grid = Grid(cfg['grid'])
    with open(os.path.join(cfg['out'], 'logs', f'seed{seed}.log'), 'w', buffering=1, encoding='utf-8') as log:
        sys.stdout = sys.stderr = log
        res = m.run_single_experiment(seed, idx0, grid, cfg['iters'], cfg['horizon'])
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    if res is None:
        return seed, None
    regrets, times = res
    df = pd.DataFrame({'iteration': np.arange(len(regrets)), 'regret': regrets,
                       'decision_time': [np.nan] + list(times)})
    df.to_csv(os.path.join(cfg['out'], f'regret_seed{seed}.csv'), index=False)
    return seed, df


def write_summary(out, frames):
    g = pd.concat(frames).groupby('iteration')
    pd.DataFrame({'regret_mean': g['regret'].mean(), 'regret_std': g['regret'].std(ddof=0),
                  'decision_time_mean': g['decision_time'].mean(), 'decision_time_std': g['decision_time'].std(ddof=0),
                  'n_seeds': g['regret'].count()}).to_csv(os.path.join(out, 'summary.csv'))


def main():
    ap = argparse.ArgumentParser(description='EARL-BO on a discrete HPO-B search space (defaults: search space 7200 of '
                                             'the paper). One process per seed; outputs regret_seed{s}.csv and '
                                             'summary.csv in --out.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--grid', default='data/hpob_7200.csv',
                    help='grid file (relative to this folder): columns X1..Xd (configuration) and y (normalised objective value)')
    ap.add_argument('--n0', type=int, default=50, help='initial design size (grid points)')
    ap.add_argument('--iters', type=int, default=50, help='decisions after the initial design (one grid point each)')
    ap.add_argument('--horizon', type=int, default=3, help='look-ahead steps per virtual episode')
    ap.add_argument('--seeds', default='0-9', help='seeds, e.g. 0-9 or 0,2,5; each uses its own initial design (0-9 only)')
    ap.add_argument('--workers', type=int, default=None, help='parallel processes; None = one per seed, at most the CPU count')
    ap.add_argument('--out', default=None, help='output folder; None = results/discrete_{grid name}_n0{n0}_h{horizon}')
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    assert all(0 <= s < N_DESIGNS for s in seeds), 'seeds must be in 0-9 (one fixed initial design each)'
    grid_path = args.grid if os.path.isabs(args.grid) else os.path.join(HERE, args.grid)
    grid = Grid(grid_path)
    assert 1 <= args.n0 and args.n0 + args.iters <= grid.n, 'n0 + iters must not exceed the number of grid points'
    name = os.path.splitext(os.path.basename(args.grid))[0]
    out = args.out or os.path.join(HERE, 'results', f'discrete_{name}_n0{args.n0}_h{args.horizon}')
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    workers = args.workers or min(len(seeds), os.cpu_count() or 1)
    designs = initial_sets(grid.n, args.n0)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(dict(vars(args), seeds=seeds, workers=workers, out=out, grid_points=grid.n, dim=grid.dim,
                       initial_sets=[d.tolist() for d in designs], started=time.strftime('%Y-%m-%d %H:%M:%S')),
                  f, indent=1)

    cfg = dict(grid=grid_path, iters=args.iters, horizon=args.horizon, out=out)
    jobs = [(s, designs[s], cfg) for s in seeds]
    print(f"EARL-BO discrete: {name} ({grid.n} points, {grid.dim}D), n0={args.n0}, {args.iters} decisions, horizon "
          f"{args.horizon}; seeds {seeds} on {workers} workers; output {out}", flush=True)
    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_worker, jobs)
    ok = {s: df for s, df in results if df is not None}
    failed = [s for s, df in results if df is None]
    if ok:
        write_summary(out, [ok[s] for s in sorted(ok)])
    print(f"done in {(time.time() - t0) / 60:.1f} min; seeds finished {sorted(ok)}, failed {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
