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
    per_alg = {a: continuous_average(m for mats in results[a].values() for m in mats)
               for a in names}
    w = np.array([1.0 if weights is None else float(weights.get(a, 0.0)) for a in names])
    if w.sum() <= 0:
        raise ValueError("Fusion weights sum to zero.")
    w = w / w.sum()
    return np.tensordot(w, np.stack([per_alg[a] for a in names]), axes=1).astype(np.float32)


# --------------------------------------------------------------------------
# Agreement weighting and the input blend
# --------------------------------------------------------------------------

def _pair_ranks(mat: np.ndarray, symmetric: bool = True) -> np.ndarray:
    """Off-diagonal scores (upper triangle of the max-symmetrised matrix when
    symmetric) as ranks scaled to [0, 1]."""
    from scipy.stats import rankdata
    n = mat.shape[0]
    if symmetric:
        v = np.maximum(mat, mat.T)[np.triu_indices(n, 1)]
    else:
        v = mat[~np.eye(n, dtype=bool)]
    return ((rankdata(v) - 1) / max(len(v) - 1, 1)).astype(np.float64)


def _from_pairs(v: np.ndarray, n: int, symmetric: bool = True) -> np.ndarray:
    out = np.zeros((n, n), dtype=np.float32)
    if symmetric:
        iu = np.triu_indices(n, 1)
        out[iu] = v; out[(iu[1], iu[0])] = v
    else:
        out[~np.eye(n, dtype=bool)] = v
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
    names = list(matrices)
    if len(names) < 3:
        return {a: 1.0 / len(names) for a in names}
    X = np.stack([_pair_ranks(matrices[a], symmetric) for a in names])
    from scipy.stats import spearmanr
    rho = np.array([spearmanr(X[i], np.delete(X, i, 0).mean(0)).statistic
                    for i in range(len(names))])
    w = np.clip(np.nan_to_num(rho), 0, None) ** power
    if w.sum() <= 0:
        w = np.ones(len(names))
    w = w / w.sum()
    return dict(zip(names, w.tolist()))


def rank_average(matrices: Dict[str, np.ndarray],
                 weights: Optional[Dict[str, float]] = None,
                 symmetric: bool = True) -> np.ndarray:
    """(Weighted) mean of each algorithm's pair ranks - the no-GCN baseline."""
    names = list(matrices)
    n = matrices[names[0]].shape[0]
    w = np.array([1.0 if weights is None else weights.get(a, 0.0) for a in names])
    w = w / w.sum()
    v = sum(wi * _pair_ranks(matrices[a], symmetric) for wi, a in zip(w, names))
    return _from_pairs(v, n, symmetric)


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
    model = _pair_ranks(fused, symmetric)
    inputs = _pair_ranks(rank_average(matrices, weights, symmetric), symmetric)
    return _from_pairs((1 - share) * model + share * inputs, n, symmetric)
