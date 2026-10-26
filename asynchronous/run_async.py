"""
Run asynchronous-batch EARL-BO on a synthetic benchmark, several seeds in parallel (one process per seed).

Defaults = the paper's 8D Ackley setting: domain [-15, 15]^8, 30 initial points, q = 3 parallel evaluations,
50 evaluations after the initial design, horizon 2, seeds 0-9. --kb on uses the Kriging Believer for the pending
evaluations (Asy EARL-BO-KB), --kb off ignores them (Asy EARL-BO); both are in the paper. Initial designs:
np.random.seed(1), then ten draws of uniform(-15, 15, (n0, dim)); seed s uses draw s, so the seeds are 0-9 and the
runs are paired across packages.

Outputs (in --out):
  regret_seed{s}.csv   iteration, regret, decision_time. Row i is the state after i finished evaluations (row 0 = the
                       initial design); regret in the minimisation units of the benchmark (optimum 0). decision_time
                       (seconds) is the decision taken at that state: row 0 carries the initial batch decision, row i
                       the replacement decision made when evaluation i finished; the last q rows have none.
  summary.csv          per iteration: mean and std (over the finished seeds) of regret and decision_time, n_seeds
  logs/seed{s}.log     stdout of seed s (training progress)
  config.json          the arguments of the run
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
FUNCTIONS = ('Ackley', 'Levy', 'Rosenbrock', 'Sum_square')
LB, UB = -15.0, 15.0
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


def initial_designs(n0, dim):
    np.random.seed(1)
    return [np.random.uniform(low=LB, high=UB, size=(n0, dim)) for _ in range(N_DESIGNS)]


def _worker(job):
    seed, X0, cfg = job
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, HERE)
    import earlbo_async as m
    with open(os.path.join(cfg['out'], 'logs', f'seed{seed}.log'), 'w', buffering=1, encoding='utf-8') as log:
        sys.stdout = sys.stderr = log
        res = m.run_single_experiment(seed, X0, cfg['func'], cfg['dim'], cfg['evals'], LB, UB, cfg['horizon'],
                                      cfg['q'], cfg['kb'])
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    if res is None:
        return seed, None
    regrets, times = res
    # times[0] = the initial batch decision (row 0), times[k] = the decision after evaluation k finished
    decision_time = list(times) + [np.nan] * (len(regrets) - len(times))
    df = pd.DataFrame({'iteration': np.arange(len(regrets)), 'regret': regrets, 'decision_time': decision_time})
    df.to_csv(os.path.join(cfg['out'], f'regret_seed{seed}.csv'), index=False)
    return seed, df


def write_summary(out, frames):
    g = pd.concat(frames).groupby('iteration')
    pd.DataFrame({'regret_mean': g['regret'].mean(), 'regret_std': g['regret'].std(ddof=0),
                  'decision_time_mean': g['decision_time'].mean(), 'decision_time_std': g['decision_time'].std(ddof=0),
                  'n_seeds': g['regret'].count()}).to_csv(os.path.join(out, 'summary.csv'))


def main():
    ap = argparse.ArgumentParser(description='Asynchronous-batch EARL-BO on a synthetic benchmark (defaults: the 8D '
                                             'Ackley experiment of the paper, q = 3, horizon 2). One process per seed; '
                                             'outputs regret_seed{s}.csv and summary.csv in --out.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--func', default='Ackley', choices=FUNCTIONS, help='benchmark (minimised; domain [-15, 15]^dim)')
    ap.add_argument('--dim', type=int, default=8, help='input dimension')
    ap.add_argument('--n0', type=int, default=30, help='initial design size')
    ap.add_argument('--q', type=int, default=3, help='parallel evaluations (= size of the initial batch)')
    ap.add_argument('--evals', type=int, default=50, help='evaluations after the initial design (q from the initial batch)')
    ap.add_argument('--horizon', type=int, default=2, help='look-ahead steps per virtual episode')
    ap.add_argument('--kb', default='off', choices=('on', 'off'),
                    help='pending evaluations: on = Kriging Believer (Asy EARL-BO-KB), off = ignored (Asy EARL-BO)')
    ap.add_argument('--seeds', default='0-9', help='seeds, e.g. 0-9 or 0,2,5; each uses its own initial design (0-9 only)')
    ap.add_argument('--workers', type=int, default=None, help='parallel processes; None = one per seed, at most the CPU count')
    ap.add_argument('--out', default=None,
                    help='output folder; None = results/async_{dim}D_{func}_h{horizon}_q{q}_kb{on|off}')
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    assert all(0 <= s < N_DESIGNS for s in seeds), 'seeds must be in 0-9 (one fixed initial design each)'
    assert 1 <= args.q <= args.evals, 'need 1 <= q <= evals'
    out = args.out or os.path.join(HERE, 'results',
                                   f'async_{args.dim}D_{args.func}_h{args.horizon}_q{args.q}_kb{args.kb}')
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    workers = args.workers or min(len(seeds), os.cpu_count() or 1)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(dict(vars(args), seeds=seeds, workers=workers, domain=[LB, UB], out=out,
                       started=time.strftime('%Y-%m-%d %H:%M:%S')), f, indent=1)

    designs = initial_designs(args.n0, args.dim)
    cfg = dict(func=args.func, dim=args.dim, evals=args.evals, horizon=args.horizon, q=args.q, kb=(args.kb == 'on'),
               out=out)
    jobs = [(s, designs[s], cfg) for s in seeds]
    print(f"EARL-BO asynchronous: {args.func} {args.dim}D, n0={args.n0}, q={args.q}, {args.evals} evaluations, horizon "
          f"{args.horizon}, kb {args.kb}; seeds {seeds} on {workers} workers; output {out}", flush=True)
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
