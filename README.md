# EARL-BO

EARL-BO (Encoder-Augmented Reinforcement Learning for Bayesian Optimization) is a multi-step look-ahead acquisition
for Bayesian optimisation. At every iteration a Gaussian-process surrogate is fitted to the data, and a PPO agent with
an attention-based DeepSets encoder is trained from scratch in a virtual environment built on that GP: the state is the
encoded data set, the action a query point, the reward the improvement that a posterior-sampled outcome brings; the
first training episodes imitate a TuRBO teacher. The mean action of the trained policy is the next real query. This
repository holds four self-contained packages: `sequential` (one point per iteration), `discrete` (the sequential
policy on an HPO-B grid), `batch` (q points per iteration from q policy heads) and `asynchronous` (q parallel
evaluations, one replacement query per finished evaluation, with or without the Kriging Believer).

## Install

Python 3.8 (tested with 3.8.19 / 3.8.20), CPU only.

```
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run

Every package is one folder and imports only from itself, so the runners can be started from the repository root. The defaults of each runner reproduce the paper's 8D Ackley
experiment (the discrete package: the HPO-B search space 7200); the other benchmark functions (`Levy`, `Rosenbrock`, `Sum_square`) are selected with `--func`, all on the
domain [-15, 15]^d. `--help` lists every option.

```
python sequential/run_sequential.py   --func Ackley --dim 8 --iters 50 --horizon 3 --seeds 0-9 --workers 10
python batch/run_batch.py             --func Ackley --dim 8 --q 3 --iters 20 --horizon 3 --seeds 0-9 --workers 10
python asynchronous/run_async.py      --func Ackley --dim 8 --q 3 --evals 50 --horizon 2 --kb off --seeds 0-9 --workers 10
python asynchronous/run_async.py      --func Ackley --dim 8 --q 3 --evals 50 --horizon 2 --kb on  --seeds 0-9 --workers 10
python discrete/run_discrete.py       --n0 50 --iters 50 --horizon 3 --seeds 0-9 --workers 10
```

| package | what one run does | defaults |
|---|---|---|
| `sequential` | one query per iteration | n0 = 30, 50 iterations, horizon 3 |
| `batch` | q queries per iteration (q policy heads, joint posterior sampling, shared team reward) | q = 3, 20 iterations, horizon 3 |
| `asynchronous` | initial batch of q from the batch policy, then one replacement query per finished evaluation from the sequential policy; `--kb on` = Kriging Believer for the pending evaluations | q = 3, 50 evaluations, horizon 2, `--kb off` |
| `discrete` | the sequential policy on the HPO-B search space 7200 (19 hyperparameters, 811 configurations, `data/hpob_7200.csv`); the answer is snapped to the nearest unevaluated grid point | n0 = 50, 50 decisions, horizon 3 |

Training schedule inside every decision: 4000 episodes of `horizon` virtual steps, one update every 50 episodes, the
first 400 episodes imitating TuRBO, the rest PPO. One process per seed with single-threaded torch/BLAS; a decision takes
a few minutes in 8D on one core, so a full 10-seed run takes hours. Seeds 0-9 each use their own fixed initial design
(synthetic benchmarks: `np.random.seed(1)` then ten uniform draws; HPO-B: `np.random.seed(0)` then ten draws of n0 grid
indices), so runs are paired across packages and with the paper.

## Output

Each run writes to `--out` (defaults: `results/seq_{dim}D_{func}_h{horizon}`, `results/batch_{dim}D_{func}_h{horizon}_q{q}`,
`results/async_{dim}D_{func}_h{horizon}_q{q}_kb{on|off}`, `results/discrete_{grid}_n0{n0}_h{horizon}`, inside the package folder):

- `regret_seed{s}.csv` — `iteration, regret, decision_time`. Row 0 is the initial design. For the synthetic benchmarks
  the regret is in the minimisation units of the function (optimum 0); for HPO-B it is the grid maximum minus the
  incumbent, with the accuracies normalised so the maximum is 1. `decision_time` is in seconds; for the asynchronous
  package row i is the state after i finished evaluations and carries the decision made at that state (row 0 the initial
  batch decision), so the last q rows have none.
- `summary.csv` — per iteration, mean and std (population, over the finished seeds) of regret and decision time, and the
  number of seeds.
- `config.json` — the arguments of the run.

## Reproducibility

All randomness is seeded (the initial designs, the GP fits, the training episodes, the policy initialisation), so a run
is bitwise reproducible on the same platform and build.

## License

`turbo.py`, `gp.py` and `utils.py` are from TuRBO (Copyright 2019 Uber Technologies, Inc.) and are licensed under the
Uber Non-Commercial License, reproduced verbatim in `LICENSE_TuRBO.txt`; `turbo.py` carries a modification notice
(`NOTICE_TuRBO.txt`). `discrete/data/hpob_7200.csv` is derived from the HPO-B meta-dataset (Pineda Arango et al.,
2021; https://github.com/releaunifreiburg/HPO-B; search space 7200, objective min-max normalised) and remains under
the HPO-B terms (MIT License, `LICENSE_HPO-B.txt`). Every other file in this repository (the EARL-BO code) is
licensed under the PolyForm Noncommercial License 1.0.0, reproduced in `LICENSE`: use for any noncommercial purpose
(research, teaching, personal study, use by non-profit and public research organisations) is permitted; commercial
use is not. Used together with the TuRBO files, the repository as a whole is in addition subject to the Uber
Non-Commercial License; where the two licenses differ, the more restrictive terms apply.

## Citation

If you use this code, please cite the EARL-BO journal article (M. Cheon, J. H. Lee, D.-Y. Koh, C. Tsay, "A Reinforcement
Learning-Based Multi-Step Lookahead Framework for Sequential, Synchronous, and Asynchronous Batch Bayesian Optimization";
bibliographic entry to be added on publication).

If you use the `discrete` package, please also cite HPO-B: S. Pineda Arango, H. S. Jomaa, M. Wistuba and J. Grabocka,
"HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML", arXiv:2106.06257, 2021.
