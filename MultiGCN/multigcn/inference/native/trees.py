"""Gradient-boosted tree importance (NONLINEARODES as GENECI ran it)."""
from __future__ import annotations

import math

import numpy as np
from joblib import Parallel, delayed


def _split_counts_sklearn(model, n_features):
    """XGBoost importance_type='weight' equivalent: how often each feature splits."""
    counts = np.zeros(n_features)
    for est in model.estimators_.ravel():
        f = est.tree_.feature
        np.add.at(counts, f[f >= 0], 1)
    s = counts.sum()
    return counts / s if s > 0 else counts


def _boost_target(X, j, n_estimators, max_depth, learning_rate, tree_method, seed):
    others = np.r_[0:j, j + 1:X.shape[1]]
    y = X[:, j]
    sd = y.std()
    y = y / sd if sd > 1e-12 else y

    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                             learning_rate=learning_rate, importance_type="weight",
                             tree_method=tree_method, base_score=0.5, n_jobs=1,
                             random_state=seed, verbosity=0)
        model.fit(X[:, others], y)
        imp = model.feature_importances_
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          learning_rate=learning_rate,
                                          random_state=seed)
        model.fit(X[:, others], y)
        imp = _split_counts_sklearn(model, len(others))

    col = np.zeros(X.shape[1])
    col[others] = imp
    return col


def nonlinearodes(X: np.ndarray, n_estimators: int = 0, max_depth: int = 5,
                  learning_rate: float = 0.1, tree_method: str = "exact",
                  seed: int = 777, n_jobs: int = -1, **_) -> np.ndarray:
    """
    Boosted-tree regulator importance - NONLINEARODES as benchmarked by GENECI.

    Despite the name, GENECI's NONLINEARODES does not fit ODEs: for each target
    it fits XGBoost (max_depth 5, n_estimators =
    round(10 * n_genes / log10(n_genes))) on the expression of all other genes
    and uses split counts ('weight' importance) as edge scores. That is what is
    reproduced here, so results stay comparable with the historical runs.

    GENECI's container pinned xgboost 0.82, whose scikit-learn wrapper defaulted
    to learning_rate=0.1 and the exact greedy tree method; modern xgboost
    defaults to 0.3 and histogram trees. The defaults here follow 0.82.

    Uses xgboost when installed (exact); otherwise scikit-learn gradient
    boosting with split-count importance (same definition, slightly different
    regulariser).

    Orientation note: GENECI's script writes this matrix TRANSPOSED (target in
    the "source" column). Here S[i, j] is regulator i -> target j like every
    other method. Irrelevant for undirected scoring, matters for directed.
    """
    n = X.shape[1]
    if n_estimators <= 0:
        n_estimators = max(10, round(n * 10 / math.log10(max(n, 10))))
    cols = Parallel(n_jobs=n_jobs)(
        delayed(_boost_target)(X, j, n_estimators, max_depth, learning_rate,
                               tree_method, seed + j)
        for j in range(n))
    S = np.column_stack(cols)
    np.fill_diagonal(S, 0.0)
    return S
