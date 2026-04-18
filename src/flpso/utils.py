import numpy as np


def rng(seed):
    return np.random.default_rng(int(seed))


def clamp(x, lb, ub):
    return np.minimum(np.maximum(x, lb), ub)


def mean_centroid_diversity(X):
    c = X.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(X - c, axis=1).mean())


class StopState:
    def __init__(self, fe_budget):
        self.fe_budget = fe_budget
        self.fe = 0
        self.best = np.inf


def eval_pop(fun, X, stop):
    vals = np.array([fun(x) for x in X])
    stop.fe += len(X)
    return vals


def pack_result(best_x, best_f, stop, best_hist, div_hist=None):
    stop.best = float(best_f)
    if div_hist is None:
        div_hist = np.array([])
    return best_x, float(best_f), stop, np.asarray(best_hist, float), np.asarray(div_hist, float)
