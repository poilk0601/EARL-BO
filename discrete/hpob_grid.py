"""
A discrete HPO-B search space as a grid of candidate points.

The grid file (data/hpob_<space>.csv) has the columns X1..Xd (the HPO-B coordinates of every configuration that was
evaluated in that search space) and y (the objective value of HPO-B, min-max normalised over the grid, so the best point has y = 1).
"""
import numpy as np
import pandas as pd
from scipy import spatial


class Grid:
    def __init__(self, path):
        df = pd.read_csv(path, float_precision='round_trip')   # the exact values of the file
        cols = [c for c in df.columns if c != 'y']
        assert cols == [f'X{i + 1}' for i in range(len(cols))] and 'y' in df.columns, list(df.columns)
        self.X = df[cols].values.astype(float)
        self.y = df['y'].values.astype(float)
        self.n, self.dim = self.X.shape
        assert len(np.unique(self.X, axis=0)) == self.n, 'duplicate grid points'
        self.lb, self.ub = self.X.min(axis=0), self.X.max(axis=0)   # the box of the Domain map, per column
        assert np.all(self.ub > self.lb), 'a constant column: the Domain box would be degenerate'
        self.y_max = float(self.y.max())

    def nearest(self, x, candidates):
        """Grid index of the candidate nearest to x (Euclidean distance in grid coordinates) and that distance."""
        idx = np.asarray(candidates, dtype=int)
        dist, local = spatial.KDTree(self.X[idx]).query(np.ravel(x))
        return int(idx[local]), float(dist)


def initial_sets(n, n0, n_sets=10):
    """The initial designs: np.random.seed(0), then n_sets draws of n0 distinct grid indices (draw s is seed s's)."""
    np.random.seed(0)
    return [np.random.choice(n, n0, replace=False) for _ in range(n_sets)]
