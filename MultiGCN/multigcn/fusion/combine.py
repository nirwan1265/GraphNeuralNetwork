"""Fusion across algorithms, iterations, and decoders."""
from __future__ import annotations

from typing import Dict, Iterable, List

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


def fuse(results: Dict[str, Dict[str, List[np.ndarray]]]) -> np.ndarray:
    """
    results[algorithm][decoder] = [reconstruction per iteration]

    Flat average across everything. Weighted, grouped, Borda-count, NMF,
    entropy-weighted and SNF alternatives were all tested; none beat a flat
    mean, and entropy-weighting + Jaccard filtering failed catastrophically.
    """
    flat = [m for per_dec in results.values()
            for mats in per_dec.values()
            for m in mats]
    return continuous_average(flat)
