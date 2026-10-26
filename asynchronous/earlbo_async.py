"""
EARL-BO, asynchronous-batch form: q evaluations run in parallel; whenever one finishes, one replacement query is chosen.

Loop. The first q queries are one decision of the synchronous-batch policy (earlbo_batch.RL_BO) on the initial design;
they complete in a random order (random.shuffle, seeded by the run's seed). Then, num_evaluations - q times, the oldest
pending evaluation finishes and joins the data, and the sequential policy (earlbo_seq.RL_BO) chooses one replacement
query that joins the end of the queue. At the end the q evaluations still pending are completed. The regret sequence has
one entry per finished evaluation (after the initial design's).

Pending evaluations. kb=False: a decision sees the finished evaluations only. kb=True: Kriging Believer. A GP fitted to
the finished evaluations gives its posterior mean at the pending points; the y scaler and the GP are fitted again on the
augmented data, and the sequential policy is trained on these. Hallucinated values never enter the real data or the
regret. Both policies run with their module defaults (schedule, learning rates, teacher).

Coordinate conventions: x through the fixed Domain map to N = [-1, 1]^d, y standardised once per decision on exactly the
rows handed to the policy; the GP is fitted by maximum likelihood with 10 restarts.
"""
import random
import time
import traceback

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import earlbo_batch as bat
import earlbo_seq as seq


def outer_fit(domain, X, y):
    """The runner step for the rows (X, y) in real units: y standardised on these rows, X mapped to N, a new GP fitted."""
    scaler_y = seq.Scaler()
    scaler_y.fit(y)
    y_scaled = scaler_y.transform(y)
    X_norm = domain.to_norm(X)
    kernel = RBF(1.0, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False)
    gpr.fit(X_norm, y_scaled)
    return gpr, scaler_y, X_norm, y_scaled


def run_single_experiment(seed, X_init, func_name, dimension, num_evaluations, domain_lb, domain_ub, horizon,
                          num_batch_actions, kb):
    """One asynchronous run with q = num_batch_actions parallel evaluations from the initial design X_init (real units)
    on `func_name` in [domain_lb, domain_ub]^dimension; num_evaluations real evaluations after the initial design.
    Returns (regrets, decision_times): regrets has num_evaluations + 1 entries (index 0 = the initial design, then one
    per finished evaluation), in the minimisation units of the benchmark (optimum 0); decision_times has
    num_evaluations - q + 1 entries (the initial batch decision first). None on error."""
    try:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)   # the completion order of the initial batch
        test_func = seq.TEST_FUNCTIONS[func_name](dimension)
        y_max_real = 0
        q = num_batch_actions
        assert isinstance(q, int) and 1 <= q <= num_evaluations, f"need 1 <= q <= num_evaluations: {q}, {num_evaluations}"

        X_train = X_init
        y_train = test_func.real(X_train).reshape(-1, 1)
        regrets = [y_max_real - np.max(y_train)]
        domain = seq.Domain([domain_lb] * dimension, [domain_ub] * dimension)
        n_decisions = num_evaluations - q + 1
        decision_times = []
        pending = []   # queued queries (real units), oldest first

        def complete():
            # The oldest pending evaluation finishes and joins the real data
            nonlocal X_train, y_train
            x_done = pending.pop(0).reshape(1, -1)
            y_done = test_func.real(x_done).reshape(-1, 1)
            X_train = np.vstack((X_train, x_done))
            y_train = np.vstack((y_train, y_done))
            regrets.append(y_max_real - np.max(y_train))

        # Phase 1: the first q queries, one decision of the synchronous-batch policy on the initial design
        gpr, scaler_y, X_norm, y_scaled = outer_fit(domain, X_train, y_train)
        batch_acq = bat.RL_BO()
        batch_acq.num_batch_actions = q
        batch_acq.horizon = horizon
        start_time = time.time()
        print(f"\n===== Decision 1/{n_decisions} (initial batch of {q}), seed {seed} =====")
        X_batch_norm = batch_acq.evaluate(gpr, X_norm, y_scaled)
        decision_times.append(time.time() - start_time)
        print(f"Decision 1/{n_decisions} took {decision_times[-1]:.1f} s")
        X_batch = domain.to_real(X_batch_norm)
        pending = [X_batch[i] for i in range(q)]
        random.shuffle(pending)

        # Phase 2: one replacement query per finished evaluation, by the sequential policy
        single_acq = seq.RL_BO()
        single_acq.horizon = horizon
        for k in range(num_evaluations - q):
            complete()
            gpr, scaler_y, X_norm, y_scaled = outer_fit(domain, X_train, y_train)
            if kb and pending:
                # Kriging Believer: the pending points get the posterior mean of the real-data GP; scaler and GP are
                # fitted again on the augmented rows
                X_pending = np.array(pending)
                mean_scaled = gpr.predict(domain.to_norm(X_pending))
                y_pending = scaler_y.inverse_transform_mean(np.asarray(mean_scaled).reshape(-1, 1))
                gpr, scaler_y, X_norm, y_scaled = outer_fit(domain, np.vstack((X_train, X_pending)),
                                                            np.vstack((y_train, y_pending)))
            start_time = time.time()
            print(f"\n===== Decision {k + 2}/{n_decisions}, seed {seed} =====")
            X_next_norm = single_acq.evaluate(gpr, X_norm, y_scaled)
            decision_times.append(time.time() - start_time)
            print(f"Decision {k + 2}/{n_decisions} took {decision_times[-1]:.1f} s")
            pending.append(domain.to_real(X_next_norm).flatten())

        while pending:   # the q evaluations still pending finish
            complete()

        return regrets, decision_times

    except Exception:
        print(f"Error in experiment with seed {seed}:")
        traceback.print_exc()
        return None
