"""Sparse Gaussian graphical model via the graphical lasso."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.covariance import graphical_lasso

from ._io import zscore


def glasso(X: np.ndarray, n_alphas: int = 12, alpha_min_ratio: float = 0.05,
           max_iter: int = 200, **_) -> np.ndarray:
    """
    Graphical lasso partial correlations, integrated over a regularisation path.

    A single glasso fit returns a sparse precision matrix: most pairs are exactly
    zero, which carries no ranking information for the top-X% topology step.
    Instead the model is fitted along a path of penalties from alpha_max (where
    every edge is zero) down to alpha_min_ratio * alpha_max, and each pair is
    scored by its mean |partial correlation| across the path. Edges that enter
    early - survive heavy shrinkage - score highest, and every pair gets a
    continuous value. Same idea as the stability-style scoring used elsewhere.

    Differs from PPCOR/GENENET: those estimate dense partial correlations, this
    one imposes sparsity, so conditional independences are enforced, not just
    estimated.
    """
    Z = zscore(X)
    emp_cov = np.cov(Z, rowvar=False, bias=True)
    n = emp_cov.shape[0]

    off = ~np.eye(n, dtype=bool)
    alpha_max = np.abs(emp_cov[off]).max()
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max * alpha_min_ratio),
                         n_alphas)[1:]

    S = np.zeros((n, n))
    fitted = 0
    for alpha in alphas:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, prec = graphical_lasso(emp_cov, alpha=alpha, max_iter=max_iter)
        except (FloatingPointError, ValueError):
            continue        # ill-conditioned at this penalty; skip it
        d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
        S += np.abs(-prec / np.outer(d, d))
        fitted += 1

    if fitted == 0:
        raise RuntimeError("graphical lasso failed to converge at every penalty.")
    S /= fitted
    np.fill_diagonal(S, 0.0)
    return S
