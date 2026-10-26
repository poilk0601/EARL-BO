"""
EARL-BO on a discrete search space (HPO-B): the sequential policy proposes a point of the box spanned by the grid, and
the nearest remaining grid point is evaluated instead.

Candidates: the grid of one HPO-B search space (hpob_grid.Grid). The initial design is a set of grid indices; an
evaluated point leaves the candidate set, so no point is queried twice. Decision: the runner step of the sequential form
(y standardised once per decision on the real rows, X mapped to N = [-1, 1]^d by the Domain map whose box is the
per-column minimum and maximum of the grid, GP fitted by maximum likelihood with 10 restarts), then
earlbo_seq.RL_BO.evaluate. Snap: the answer, mapped back to grid coordinates, is replaced by the nearest remaining grid
point (Euclidean distance, scipy KDTree); that point is evaluated and joins the data. Regret: grid maximum minus the
incumbent (0 once the best grid point has been found).
"""
import time
import traceback

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import earlbo_seq as seq


def run_single_experiment(seed, initial_indices, grid, num_decisions, horizon):
    """One run on `grid` (hpob_grid.Grid) from the initial design `initial_indices` (grid indices).
    Returns (regrets, decision_times): regrets has num_decisions + 1 entries (index 0 = the initial design);
    decision_times has one entry per decision (seconds). None on error."""
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        y_max_real = grid.y_max

        initial_indices = np.asarray(initial_indices, dtype=int)
        assert len(np.unique(initial_indices)) == len(initial_indices), 'repeated initial index'
        remaining = np.ones(grid.n, dtype=bool)   # the candidate set
        remaining[initial_indices] = False
        X_train = grid.X[initial_indices]
        y_train = grid.y[initial_indices].reshape(-1, 1)
        regrets = [y_max_real - np.max(y_train)]

        domain = seq.Domain(grid.lb, grid.ub)
        kernel = RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False)
        scaler_y = seq.Scaler()
        acq_func = seq.RL_BO()
        acq_func.horizon = horizon
        decision_times = []

        for j in range(num_decisions):
            scaler_y.fit(y_train)
            y_train_scaled = scaler_y.transform(y_train)
            X_train_norm = domain.to_norm(X_train)
            gpr.fit(X_train_norm, y_train_scaled)

            start_time = time.time()
            print(f"\n===== Decision {j + 1}/{num_decisions}, seed {seed} =====")
            X_next_norm = acq_func.evaluate(gpr, X_train_norm, y_train_scaled)
            decision_times.append(time.time() - start_time)
            print(f"Decision {j + 1}/{num_decisions} took {decision_times[-1]:.1f} s")

            # The nearest remaining grid point replaces the answer and leaves the candidate set
            x_answer = domain.to_real(X_next_norm).ravel()
            candidates = np.flatnonzero(remaining)
            index_next, dist = grid.nearest(x_answer, candidates)
            remaining[index_next] = False
            print(f"Snapped to grid point {index_next} (distance {dist:.3g}), y = {grid.y[index_next]:.6g}")
            X_train = np.vstack((X_train, grid.X[index_next].reshape(1, -1)))
            y_train = np.vstack((y_train, grid.y[index_next].reshape(1, 1)))
            regrets.append(y_max_real - np.max(y_train))

        return regrets, decision_times

    except Exception:
        print(f"Error in experiment with seed {seed}:")
        traceback.print_exc()
        return None
