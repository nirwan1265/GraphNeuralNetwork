"""
Shared input/output for MultiGCN's native Python algorithms.

Every native algorithm is a pure function returning an (n_genes, n_genes)
score matrix S where S[i, j] is the confidence of a regulatory link from gene i
(regulator) to gene j (target). These helpers handle the parts that are the
same for all of them: reading the expression and time-series formats the rest
of the package uses, and writing the GRN_<NAME>.csv edge-list contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def read_expression(path) -> Tuple[np.ndarray, List[str]]:
    """
    Canonical input: GENES x SAMPLES CSV, gene ids in column 1.

    Returns (X, genes) with X as SAMPLES x GENES - the orientation every
    statistical routine below expects.
    """
    df = pd.read_csv(path, index_col=0)
    return df.to_numpy(dtype=np.float64).T, [str(g) for g in df.index]


def read_time_series(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Time-series input: first column 'time', remaining columns genes, one row
    per measurement, experiments concatenated. A time value that does not
    increase marks the start of the next experiment.

    Returns (X, times, experiment_id, genes) with X as MEASUREMENTS x GENES.
    """
    df = pd.read_csv(path)
    tcol = next((c for c in df.columns if str(c).lower() == "time"), None)
    if tcol is None:
        raise ValueError(f"{path} has no 'time' column; time-series input required.")
    times = df[tcol].to_numpy(dtype=np.float64)
    genes = [str(c) for c in df.columns if c != tcol]
    X = df[genes].to_numpy(dtype=np.float64)

    exp = np.zeros(len(times), dtype=int)
    for k in range(1, len(times)):
        exp[k] = exp[k - 1] + (1 if times[k] <= times[k - 1] else 0)
    return X, times, exp, genes


def zscore(X: np.ndarray, ddof: int = 0) -> np.ndarray:
    """Column-wise z-score. Constant columns become zeros instead of NaN."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=ddof)
    sd[sd < 1e-12] = 1.0
    return (X - mu) / sd


def write_edge_list(S: np.ndarray, genes: List[str], out_dir, name: str) -> Path:
    """
    Write S as GRN_<name>.csv: headerless (source, target, score), self-edges
    removed, scores min-max scaled to [0, 1] over off-diagonal entries, zeros
    dropped, sorted descending. Identical contract to every other wrapper.
    """
    S = np.asarray(S, dtype=np.float64).copy()
    if S.shape != (len(genes), len(genes)):
        raise ValueError(f"{name}: score matrix {S.shape} does not match "
                         f"{len(genes)} genes")
    np.fill_diagonal(S, np.nan)
    S[~np.isfinite(S)] = np.nan

    off = ~np.isnan(S)
    if not off.any():
        raise ValueError(f"{name} produced no finite scores.")
    lo, hi = np.nanmin(S), np.nanmax(S)
    S = (S - lo) / (hi - lo) if hi - lo > 1e-15 else np.where(off, 0.0, np.nan)

    i, j = np.nonzero(off & (S > 0))
    df = pd.DataFrame({"source": np.array(genes)[i],
                       "target": np.array(genes)[j],
                       "score": S[i, j]})
    df = df.sort_values("score", ascending=False)

    path = Path(out_dir) / f"GRN_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, header=False, index=False)
    return path
