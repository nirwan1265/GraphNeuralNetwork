"""
Time-series methods: SINCERITIES and SCODE.

Both were designed for single-cell snapshots ordered in time. With bulk time
courses of several perturbation experiments (DREAM4: 10 experiments x 21
time points), the measurements taken at the same time point across
experiments play the role of the cells captured at that time point.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import nnls
from scipy.stats import ks_2samp


# --------------------------------------------------------------------------
# SINCERITIES (Papili Gao et al. 2018, Bioinformatics 34:258)
# --------------------------------------------------------------------------

def distribution_distances(X: np.ndarray, times: np.ndarray,
                           normalize_dt: bool = True) -> np.ndarray:
    """
    DD[l, g] = Kolmogorov-Smirnov distance between gene g's expression
    distribution at time point l and at l + 1, divided by the time step.
    """
    tps = np.unique(times)
    if len(tps) < 3:
        raise ValueError("SINCERITIES needs at least 3 distinct time points.")
    groups = [X[times == t] for t in tps]
    DD = np.zeros((len(tps) - 1, X.shape[1]))
    for l in range(len(tps) - 1):
        a, b = groups[l], groups[l + 1]
        for g in range(X.shape[1]):
            DD[l, g] = ks_2samp(a[:, g], b[:, g]).statistic
        if normalize_dt:
            DD[l] /= (tps[l + 1] - tps[l])
    return DD


def _nn_ridge(Xs, y, lam):
    """Non-negative ridge: min ||y - Xb||^2 + lam ||b||^2, b >= 0 (via NNLS)."""
    p = Xs.shape[1]
    A = np.vstack([Xs, np.sqrt(lam) * np.eye(p)])
    rhs = np.concatenate([y, np.zeros(p)])
    return nnls(A, rhs, maxiter=50 * p)[0]


def _sincerities_target(P, y, lambdas):
    """glmnet-style: centre y, standardise predictors, LOO-CV over lambda."""
    mu_x, sd_x = P.mean(0), P.std(0)
    sd_x[sd_x < 1e-12] = 1.0
    Xs = (P - mu_x) / sd_x
    yc = y - y.mean()
    m = len(y)

    best_lam, best_err = lambdas[0], np.inf
    for lam in lambdas:
        err = 0.0
        for k in range(m):
            keep = np.r_[0:k, k + 1:m]
            b = _nn_ridge(Xs[keep], yc[keep], lam)
            err += (yc[k] - Xs[k] @ b) ** 2
        if err < best_err:
            best_lam, best_err = lam, err
    return _nn_ridge(Xs, yc, best_lam) / sd_x       # back to original scale


def sincerities(X: np.ndarray, times: np.ndarray, normalize_dt: bool = True,
                n_lambdas: int = 20, n_jobs: int = -1, **_) -> np.ndarray:
    """
    SINCERITIES: regress each target's distribution change on every gene's
    distribution change one time step earlier.

        DD_target(l + 1) = sum_g alpha_g * DD_g(l),   alpha >= 0

    fitted by non-negative ridge regression with leave-one-out choice of the
    penalty, as in the original (glmnet, alpha = 0, lower.limits = 0). The
    score for regulator g -> target j is alpha_g in target j's model.
    """
    DD = distribution_distances(X, times, normalize_dt)
    P, R = DD[:-1], DD[1:]                          # predictors at l, response at l+1
    lambdas = np.logspace(-3, 2, n_lambdas) * len(P)
    cols = Parallel(n_jobs=n_jobs)(
        delayed(_sincerities_target)(P, R[:, j], lambdas) for j in range(X.shape[1]))
    S = np.column_stack(cols)
    np.fill_diagonal(S, 0.0)
    return np.abs(S)


# --------------------------------------------------------------------------
# SCODE (Matsumoto et al. 2017, Bioinformatics 33:2314)
# --------------------------------------------------------------------------

def _scode_run(X, t, D, n_iter, b_min, b_max, rng):
    """
    One SCODE optimisation: model x(t) = W z(t), z_d(t) = exp(b_d t), and
    random-search the diagonal B, solving W by least squares at each step.
    Returns A = W diag(b) W^+, the linear ODE dx/dt = A x.
    """
    def fit(b):
        Z = np.exp(np.outer(b, t))                   # (D, M)
        W = np.linalg.lstsq(Z.T, X, rcond=None)[0].T  # (G, D)
        rss = np.sum((X.T - W @ Z) ** 2)
        return rss, W

    b = rng.uniform(b_min, b_max, D)
    best_rss, best_W = fit(b)
    for _ in range(n_iter):
        cand = b.copy()
        cand[rng.integers(D)] = rng.uniform(b_min, b_max)
        rss, W = fit(cand)
        if rss < best_rss:
            b, best_rss, best_W = cand, rss, W
    return best_W @ np.diag(b) @ np.linalg.pinv(best_W)


def scode(X: np.ndarray, times: np.ndarray, n_latent: int = 4, n_iter: int = 100,
          n_runs: int = 20, b_min: float = -10.0, b_max: float = 2.0,
          seed: int = 777, **_) -> np.ndarray:
    """
    SCODE: a linear ODE dx/dt = A x fitted through a low-dimensional latent
    trajectory. Defaults follow the paper (D = 4 latent dimensions, B sampled
    in [-10, 2], 100 random-search iterations); A is averaged over n_runs
    independent optimisations, as the authors recommend, because a single run
    is noisy.

    dx_i/dt = sum_j A_ij x_j, so regulator j -> target i is A[i, j]; the
    returned score matrix is |A| transposed into regulator-row orientation.
    Time is rescaled to [0, 1], SCODE's pseudotime convention.
    """
    t = (times - times.min()) / max(times.max() - times.min(), 1e-12)
    rng = np.random.default_rng(seed)
    A = np.zeros((X.shape[1], X.shape[1]))
    for _ in range(n_runs):
        A += _scode_run(X, t, n_latent, n_iter, b_min, b_max, rng)
    S = np.abs(A / n_runs).T
    np.fill_diagonal(S, 0.0)
    return S
