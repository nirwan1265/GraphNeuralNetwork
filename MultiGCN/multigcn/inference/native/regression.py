"""Linear regression methods: elastic net and PLSNET."""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import enet_path

from ._io import zscore


# --------------------------------------------------------------------------
# Elastic net
# --------------------------------------------------------------------------

def _enet_target(Z, j, l1_ratio, n_alphas, eps, candidates=None):
    others = (np.r_[0:j, j + 1:Z.shape[1]] if candidates is None
              else np.array([c for c in candidates if c != j]))
    Xo, y = Z[:, others], Z[:, j]
    # Explicit grid (largest penalty that still zeroes every coefficient, down
    # to eps of it). Passing the array works across scikit-learn versions,
    # whose integer-grid argument changed name in 1.9.
    alpha_max = np.abs(Xo.T @ y).max() / (len(y) * max(l1_ratio, 1e-3))
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max * eps), n_alphas)
    _, coefs, _ = enet_path(Xo, y, l1_ratio=l1_ratio, alphas=alphas)
    col = np.zeros(Z.shape[1])
    col[others] = np.abs(coefs).mean(axis=1)
    return col


def elasticnet(X: np.ndarray, l1_ratio: float = 0.5, n_alphas: int = 50,
               eps: float = 1e-3, n_jobs: int = -1, regulators=None,
               **_) -> np.ndarray:
    """
    Per-target elastic net, integrated over the regularisation path.

    For each target gene, regress it on all other genes (standardised) along a
    path of penalties and score each regulator by its mean |coefficient| over
    the path. Regulators that enter at strong penalty and stay large score
    highest; every pair gets a continuous score, unlike a single sparse fit.

    A linear, additive bias - deliberately different from the tree ensembles
    (GENIE3 and friends), which capture interactions but not linear structure
    as efficiently.

    regulators: optional candidate-regulator indices (e.g. TFs); only they
    are used as predictors, so only their rows can score.
    """
    Z = zscore(X)
    cols = Parallel(n_jobs=n_jobs)(
        delayed(_enet_target)(Z, j, l1_ratio, n_alphas, eps, regulators)
        for j in range(Z.shape[1]))
    S = np.column_stack(cols)          # S[:, j] = regulators' scores for target j
    np.fill_diagonal(S, 0.0)
    return S


# --------------------------------------------------------------------------
# PLSNET (Guo et al. 2016, BMC Bioinformatics 17:293)
#
# Faithful port of the reference MATLAB (plsnet.m / plsrank.m, as shipped by
# GENECI): T bootstrap rounds; in each, every target is regressed on all other
# z-scored genes with univariate-response SIMPLS, regulators are ranked by the
# paper's VIP formula, and the summed importances are finally reweighted by
# the variance of each regulator's out-going scores.
# --------------------------------------------------------------------------

def _zscore_ddof1(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)        # MATLAB zscore uses N-1
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def _simpls_vip_batched(Xb: np.ndarray, Y: np.ndarray, ncomp: int,
                        dx: int) -> np.ndarray:
    """
    SIMPLS for many univariate regressions at once.

    Xb : (n_targets, N, p)  predictors per target (target's own column zeroed,
                            which is exactly equivalent to dropping it)
    Y  : (n_targets, N)     responses
    Returns VIP (n_targets, p) using plsrank.m's formula:
        vip = dx * pctvarY @ (W .^ 2)' / sum(pctvarY)
    """
    nt_, N, p = Xb.shape
    X0 = Xb - Xb.mean(axis=1, keepdims=True)
    y0 = Y - Y.mean(axis=1, keepdims=True)

    V = np.zeros((nt_, p, ncomp))
    W = np.zeros((nt_, p, ncomp))
    Q = np.zeros((nt_, ncomp))
    cov = np.einsum("jnp,jn->jp", X0, y0)

    for i in range(ncomp):
        s = np.linalg.norm(cov, axis=1)
        s = np.where(s < 1e-12, 1e-12, s)
        r = cov / s[:, None]                         # first left singular vector
        t = np.einsum("jnp,jp->jn", X0, r)
        normt = np.linalg.norm(t, axis=1)
        normt = np.where(normt < 1e-12, 1e-12, normt)
        t /= normt[:, None]
        xload = np.einsum("jnp,jn->jp", X0, t)
        Q[:, i] = s / normt                          # y-loading (dy = 1)
        W[:, :, i] = r / normt[:, None]

        v = xload
        if i > 0:
            Vi = V[:, :, :i]
            for _ in range(2):                       # re-orthogonalise twice
                v = v - np.einsum("jpk,jk->jp", Vi, np.einsum("jpk,jp->jk", Vi, v))
        v = v / np.maximum(np.linalg.norm(v, axis=1), 1e-12)[:, None]
        V[:, :, i] = v
        cov = cov - v * np.sum(v * cov, axis=1, keepdims=True)
        Vi = V[:, :, :i + 1]
        cov = cov - np.einsum("jpk,jk->jp", Vi, np.einsum("jpk,jp->jk", Vi, cov))

    pct_y = Q ** 2 / np.maximum((y0 ** 2).sum(axis=1), 1e-12)[:, None]
    denom = np.maximum(pct_y.sum(axis=1), 1e-12)
    return dx * np.einsum("jpk,jk->jp", W ** 2, pct_y) / denom[:, None]


def _plsnet_rounds(X, n_rounds, ncomp, seed):
    rng = np.random.default_rng(seed)
    N, p = X.shape
    total = np.zeros((p, p))                         # [target, regulator]
    diag = np.arange(p)
    for _ in range(n_rounds):
        Zb = _zscore_ddof1(X[rng.integers(0, N, N)])
        Xb = np.broadcast_to(Zb, (p, N, p)).copy()
        Xb[diag, :, diag] = 0.0          # target j never predicts itself
        total += _simpls_vip_batched(Xb, Zb.T, ncomp, dx=p - 1)
    return total


def plsnet(X: np.ndarray, n_components: int = 5, n_iter: int = 1000,
           seed: int = 777, n_jobs: int = -1, **_) -> np.ndarray:
    """
    PLSNET: bootstrap-aggregated PLS variable importance (VIP).

    Defaults match the reference run (5 components, 1000 bootstrap rounds, all
    genes as candidate regulators). The final refinement multiplies each
    regulator's scores by the variance of that regulator's out-going scores,
    which favours hubs whose influence is target-specific rather than uniform.
    """
    from joblib import cpu_count
    workers = cpu_count() if n_jobs in (-1, 0) else max(1, n_jobs)
    workers = max(1, min(workers, n_iter))
    chunks = np.array_split(np.arange(n_iter), workers)
    seeds = np.random.SeedSequence(seed).spawn(len(chunks))

    parts = Parallel(n_jobs=workers)(
        delayed(_plsnet_rounds)(X, len(c), n_components, s.generate_state(1)[0])
        for c, s in zip(chunks, seeds) if len(c))
    vim = sum(parts).T                               # [regulator, target]

    theta = vim.var(axis=1, ddof=1)
    vim = vim * theta[:, None]
    np.fill_diagonal(vim, 0.0)
    return vim
