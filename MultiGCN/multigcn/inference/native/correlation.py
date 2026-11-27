"""Co-expression methods: Pearson/Spearman correlation and WGCNA."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _corr(X: np.ndarray, method: str = "pearson") -> np.ndarray:
    if method == "spearman":
        X = np.apply_along_axis(rankdata, 0, X)
    elif method != "pearson":
        raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
    C = np.corrcoef(X, rowvar=False)
    return np.nan_to_num(C, nan=0.0)


def pearson(X: np.ndarray, method: str = "pearson", **_) -> np.ndarray:
    """
    |correlation| between every gene pair.

    The simplest co-expression baseline, and the one reviewers expect to see.
    Undirected by construction: S is symmetric.
    """
    S = np.abs(_corr(X, method))
    np.fill_diagonal(S, 0.0)
    return S


# --------------------------------------------------------------------------
# WGCNA (Langfelder & Horvath 2008)
# --------------------------------------------------------------------------

_WGCNA_POWERS = list(range(1, 11)) + list(range(12, 21, 2))


def _scale_free_r2(k: np.ndarray, n_breaks: int = 10) -> float:
    """
    WGCNA's scaleFreeFitIndex: bin connectivity into n_breaks equal-width bins,
    regress log10(p(k)) on log10(mean k per bin), return the signed R^2
    (-sign(slope) * R^2), so a proper scale-free decay gives a positive value.
    """
    edges = np.linspace(k.min(), k.max(), n_breaks + 1)
    idx = np.clip(np.digitize(k, edges[1:-1]), 0, n_breaks - 1)
    dk, pk = [], []
    for b in range(n_breaks):
        sel = idx == b
        if sel.any():
            dk.append(k[sel].mean())
            pk.append(sel.mean())
    dk, pk = np.array(dk), np.array(pk)
    keep = (dk > 0) & (pk > 0)
    if keep.sum() < 3:
        return 0.0
    x, y = np.log10(dk[keep]), np.log10(pk[keep])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
    return float(-np.sign(slope) * r2)


def pick_soft_threshold(C: np.ndarray, network_type: str = "unsigned",
                        rsquared_cut: float = 0.85) -> int:
    """Lowest power whose signed scale-free R^2 exceeds the cut (WGCNA default 0.85)."""
    base = np.abs(C) if network_type == "unsigned" else (1 + C) / 2
    for beta in _WGCNA_POWERS:
        A = base ** beta
        np.fill_diagonal(A, 0.0)
        if _scale_free_r2(A.sum(axis=1)) > rsquared_cut:
            return beta
    # WGCNA's own guidance when no power reaches the cut on unsigned networks
    # with this many samples.
    return 6


def wgcna(X: np.ndarray, beta: int = 0, network_type: str = "unsigned",
          rsquared_cut: float = 0.85, **_) -> np.ndarray:
    """
    WGCNA topological overlap (TOM) as edge confidence.

    adjacency a_ij = |cor_ij|^beta (unsigned) or ((1 + cor_ij)/2)^beta (signed)
    TOM_ij = (sum_u a_iu a_uj + a_ij) / (min(k_i, k_j) + 1 - a_ij)

    TOM scores a pair highly when the two genes share neighbours, not just when
    they correlate, which damps spurious isolated correlations. beta=0 picks the
    soft threshold automatically the way WGCNA's pickSoftThreshold does.
    Only the adjacency/TOM core is needed here; module detection is not.
    """
    C = _corr(X, "pearson")
    if beta <= 0:
        beta = pick_soft_threshold(C, network_type, rsquared_cut)
    base = np.abs(C) if network_type == "unsigned" else (1 + C) / 2
    A = base ** beta
    np.fill_diagonal(A, 0.0)

    k = A.sum(axis=1)
    shared = A @ A
    denom = np.minimum.outer(k, k) + 1.0 - A
    TOM = (shared + A) / denom
    np.fill_diagonal(TOM, 0.0)
    return TOM
