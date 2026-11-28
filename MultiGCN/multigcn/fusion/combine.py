"""Fusion across algorithms, iterations, and decoders."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


def minmax(mat: np.ndarray) -> np.ndarray:
    lo, hi = float(mat.min()), float(mat.max())
    if hi - lo < 1e-12:
        return np.zeros_like(mat, dtype=np.float32)
    return ((mat - lo) / (hi - lo)).astype(np.float32)


def continuous_average(reconstructions: Iterable[np.ndarray]) -> np.ndarray:
    """
    Element-wise mean of every [0,1]-normalised reconstruction.

    Do NOT binarise before combining. Binarise-then-vote-count (the original
    method) was tested against this on identical trained models, changing only
    the combination step, and lost ~0.02 AUROC. Early binarisation discards
    exactly how confident each prediction was.
    """
    mats = [minmax(np.asarray(m, dtype=np.float32)) for m in reconstructions]
    if not mats:
        raise ValueError("No reconstructions to fuse.")
    return np.mean(np.stack(mats, axis=0), axis=0).astype(np.float32)


def fuse(results: Dict[str, Dict[str, List[np.ndarray]]],
         weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    results[algorithm][decoder] = [reconstruction per iteration]

    Each algorithm's runs (every decoder, every iteration) are min-max scaled
    and averaged; algorithms are then combined with `weights` (normalised to
    sum to 1), or equally if weights is None - identical to a flat mean over
    all runs when every algorithm has the same number of runs.

    Grouping algorithms by family, Borda count, NMF, entropy weighting and SNF
    were all tested and none beat a flat mean; agreement weighting
    (agreement_weights) is the one scheme that did.
    """
    names = list(results)
    if not names:
        raise ValueError("No reconstructions to fuse.")
    w = np.array([1.0 if weights is None else float(weights.get(a, 0.0)) for a in names])
    if w.sum() <= 0:
        raise ValueError("Fusion weights sum to zero.")
    w = w / w.sum()
    out = None                     # running sum: never more than one extra n x n
    for wi, a in zip(w, names):
        if wi <= 0:
            continue
        m = continuous_average(m for mats in results[a].values() for m in mats)
        m *= wi
        out = m if out is None else out + m
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# Agreement weighting and the input blend
# --------------------------------------------------------------------------

def _support(matrices: Dict[str, np.ndarray], symmetric: bool = True) -> np.ndarray:
    """Pair indices (flat, into the upper triangle or the off-diagonal) that at
    least one algorithm scored above zero. Pairs nobody scored tie at the
    bottom of every ranking, so ranking only the support gives the same order
    at a fraction of the memory - on DREAM5 E. coli with TF-restricted inputs,
    ~1.5M of 10M pairs."""
    names = list(matrices)
    n = matrices[names[0]].shape[0]
    sel = np.triu_indices(n, 1) if symmetric else np.nonzero(~np.eye(n, dtype=bool))
    any_pos = np.zeros(len(sel[0]), dtype=bool)
    for a in names:
        M = matrices[a]
        v = np.maximum(M[sel], M.T[sel]) if symmetric else M[sel]
        any_pos |= v > 0
    return sel[0][any_pos], sel[1][any_pos]


def _pair_ranks(mat: np.ndarray, symmetric: bool = True, support=None) -> np.ndarray:
    """Scores over `support` pairs (default: every upper-triangle pair when
    symmetric, else every off-diagonal pair) as ranks scaled to [0, 1]."""
    from scipy.stats import rankdata
    n = mat.shape[0]
    if support is None:
        support = np.triu_indices(n, 1) if symmetric else np.nonzero(~np.eye(n, dtype=bool))
    i, j = support
    v = np.maximum(mat[i, j], mat[j, i]) if symmetric else mat[i, j]
    return ((rankdata(v) - 1) / max(len(v) - 1, 1)).astype(np.float32)


def _from_pairs(v: np.ndarray, n: int, symmetric: bool = True, support=None) -> np.ndarray:
    out = np.zeros((n, n), dtype=np.float32)
    if support is None:
        support = np.triu_indices(n, 1) if symmetric else np.nonzero(~np.eye(n, dtype=bool))
    i, j = support
    out[i, j] = v
    if symmetric:
        out[j, i] = v
    return out


def agreement_weights(matrices: Dict[str, np.ndarray], power: float = 2.0,
                      symmetric: bool = True) -> Dict[str, float]:
    """
    Weight each algorithm by how well it agrees with the OTHER algorithms.

    rho_a = Spearman correlation between algorithm a's scores and the mean
    rank of every other algorithm; weight_a = max(rho_a, 0) ** power,
    normalised to sum to 1. power=0 is the flat average.

    Uses no gold standard. On DREAM4 the algorithms that stand apart from the
    rest (LEAP, NONLINEARODES, MEOMI: rho 0.12-0.20 on network 1) are mostly
    noise, while the strongest ones (GENIE3, TIGRESS, PEARSON) sit near the
    consensus. Weighting by agreement beat the flat average on all five DREAM4
    networks; grouping algorithms by family, which up-weights the loners, was
    worse on all five.
    """
    from scipy.stats import rankdata
    names = list(matrices)
    if len(names) < 3:
        return {a: 1.0 / len(names) for a in names}
    sup = _support(matrices, symmetric)
    X = np.stack([_pair_ranks(matrices[a], symmetric, sup) for a in names])
    total = X.sum(axis=0, dtype=np.float64)
    R = len(names)
    rho = np.empty(R)
    for k in range(R):                     # Spearman = Pearson on ranks
        a = X[k].astype(np.float64)        # (already ranks)
        b = rankdata((total - a) / (R - 1))
        a = a - a.mean(); b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        rho[k] = (a * b).sum() / den if den > 0 else 0.0
    w = np.clip(np.nan_to_num(rho), 0, None) ** power
    if w.sum() <= 0:
        w = np.ones(R)
    w = w / w.sum()
    return dict(zip(names, w.tolist()))


def rank_average(matrices: Dict[str, np.ndarray],
                 weights: Optional[Dict[str, float]] = None,
                 symmetric: bool = True) -> np.ndarray:
    """(Weighted) mean of each algorithm's pair ranks - the no-GCN baseline."""
    names = list(matrices)
    n = matrices[names[0]].shape[0]
    sup = _support(matrices, symmetric)
    w = np.array([1.0 if weights is None else weights.get(a, 0.0) for a in names])
    w = w / w.sum()
    v = np.zeros(len(sup[0]), dtype=np.float64)
    for wi, a in zip(w, names):
        if wi > 0:
            v += wi * _pair_ranks(matrices[a], symmetric, sup)
    return _from_pairs(v.astype(np.float32), n, symmetric, sup)


def blend_with_inputs(fused: np.ndarray, matrices: Dict[str, np.ndarray],
                      share: float = 0.5,
                      weights: Optional[Dict[str, float]] = None,
                      symmetric: bool = True) -> np.ndarray:
    """
    (1 - share) * rank(model output) + share * rank((weighted) rank-average).

    The model is strongest at the top of the ranking (AUPR), the rank-average
    of the raw scores across the whole ranking (AUROC); a 50/50 rank blend
    keeps both. share=0 returns the model output unchanged.
    """
    if share <= 0:
        return fused
    if not 0 < share <= 1:
        raise ValueError(f"input blend share must be in [0, 1], got {share}")
    n = fused.shape[0]
    model = _pair_ranks(fused, symmetric)            # every pair: the model scores all
    inputs = _pair_ranks(rank_average(matrices, weights, symmetric), symmetric)
    return _from_pairs((1 - share) * model + share * inputs, n, symmetric)
