"""
Algorithms that use gene knockout experiments.

A knockout is interventional evidence: if knocking out gene i shifts gene j far
from its usual level, i is upstream of j. On DREAM4 this is the strongest
signal there is - the knockout Z-score below, a single formula, scores
roughly twice the AUPR of any expression-only method.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def ko_zscore(X: np.ndarray, X_ko: np.ndarray, ko_index: List[int],
              signed: bool = False, **_) -> np.ndarray:
    """
    Null-mutant Z-score (Prill et al. 2010), as PORTIA computes it.

        Z[i, j] = |x_j(knockout of i) - mean_j| / sd_j

    mean_j and sd_j are taken over all knockout experiments - in a set of
    single knockouts most genes are unaffected by most knockouts, so that is
    the background. Repeated knockouts of the same gene are averaged. Genes
    that were never knocked out get no outgoing score (row of zeros).

    X (the ordinary expression data) is unused; it is accepted so every
    native algorithm has the same call shape.
    """
    n = X_ko.shape[1]
    mu = X_ko.mean(axis=0)
    sd = X_ko.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = np.zeros((n, n))
    counts = np.zeros(n)
    for row, i in zip(X_ko, ko_index):
        z = (row - mu) / sd
        Z[i] += z if signed else np.abs(z)
        counts[i] += 1
    hit = counts > 0
    Z[hit] /= counts[hit, None]
    np.fill_diagonal(Z, 0.0)
    if signed:
        Z = np.abs(Z)       # direction of change kept only through averaging
    return Z


def portia(X: np.ndarray, X_ko: Optional[np.ndarray] = None,
           ko_index: Optional[List[int]] = None, method: str = "fast",
           lambda1: float = 0.8, normalize: bool = True, **_) -> np.ndarray:
    """
    PORTIA (Passemiers, Moreau & Raimondi 2022, Bioinformatics 38:2802):
    power transform, shrunk precision matrix, gene-bias correction, then
    direction from regression and - when knockouts are given - the knockout
    Z-scores. Wraps the authors' package (portia-grn).

    X are all non-knockout samples (wild type, knockdowns, time points ...);
    knockout experiments are added and flagged so PORTIA can use them.
    Without knockouts it still runs, on expression alone.
    """
    try:
        import portia as pt
    except ImportError as e:  # pragma: no cover
        raise ImportError("PORTIA needs the portia-grn package: "
                          "pip install portia-grn") from e
    import warnings

    ds = pt.GeneExpressionDataset()
    for k, row in enumerate(X):
        ds.add(pt.Experiment(k, row))
    if X_ko is not None:
        for k, (row, i) in enumerate(zip(X_ko, ko_index)):
            ds.add(pt.Experiment(len(X) + k, row, knockout=[i]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # "negative values" notice: handled by PORTIA
        S = pt.run(ds, method=method, lambda1=lambda1, normalize=normalize,
                   verbose=False)
    S = np.asarray(S, dtype=np.float64)
    np.fill_diagonal(S, 0.0)
    return S
