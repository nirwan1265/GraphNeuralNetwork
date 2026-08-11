"""Loading per-algorithm edge lists into dense score matrices."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataError(RuntimeError):
    pass


def read_expression(path) -> pd.DataFrame:
    """
    Canonical input format: GENES x SAMPLES.
    Row index = gene ids, header row = sample ids.
    """
    df = pd.read_csv(path, index_col=0)
    if df.shape[0] < 2 or df.shape[1] < 2:
        raise DataError(f"Expression matrix at {path} has shape {df.shape}; "
                        "expected genes as rows and samples as columns.")
    if df.isna().any().any():
        raise DataError(f"Expression matrix at {path} contains NaNs.")
    return df


def load_edge_list(path, genes: List[str]) -> np.ndarray:
    """
    Read one headerless 3-column edge list (source, target, confidence) and
    expand it to a dense (n_genes, n_genes) matrix. Unlisted pairs are 0.
    """
    path = Path(path)
    df = pd.read_csv(path, header=None, names=["source", "target", "score"])
    if df.empty:
        raise DataError(f"{path.name} is empty.")

    index = {g: i for i, g in enumerate(genes)}
    unknown = set(df["source"]) | set(df["target"]) - set(index)
    unknown = {g for g in unknown if g not in index}
    if unknown:
        raise DataError(
            f"{path.name} references {len(unknown)} gene(s) absent from the "
            f"expression matrix, e.g. {sorted(unknown)[:5]}. "
            "This usually means the edge list and expression file come from "
            "different datasets."
        )

    n = len(genes)
    mat = np.zeros((n, n), dtype=np.float32)
    rows = df["source"].map(index).to_numpy()
    cols = df["target"].map(index).to_numpy()
    mat[rows, cols] = df["score"].to_numpy(dtype=np.float32)

    # Guard against the pre-binarised-file trap documented in the project notes:
    # a "confidence" column with only two distinct values has no gradation left
    # to threshold, and every downstream sweep on it is meaningless.
    distinct = np.unique(mat[mat != 0])
    if distinct.size <= 2 and n > 10:
        raise DataError(
            f"{path.name} has only {distinct.size} distinct non-zero value(s) "
            f"({distinct.tolist()}). This looks pre-binarised, not continuous. "
            "Use the raw GRN_<ALGORITHM>.csv output, never a "
            "normalized_weight_matrix_*.csv file."
        )
    return mat


def load_algorithm_matrices(
    lists_dir,
    genes: List[str],
    methods: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load every GRN_<ALGORITHM>.csv in `lists_dir` into a dict of dense matrices.
    """
    lists_dir = Path(lists_dir)
    if not lists_dir.is_dir():
        raise DataError(f"{lists_dir} is not a directory.")

    found = sorted(lists_dir.glob("GRN_*.csv"))
    if not found:
        raise DataError(
            f"No GRN_*.csv files in {lists_dir}. "
            "Did the inference step actually run?"
        )

    out: Dict[str, np.ndarray] = {}
    for f in found:
        name = f.stem[len("GRN_"):]
        if methods is not None and name not in methods:
            continue
        out[name] = load_edge_list(f, genes)

    if methods is not None:
        missing = [m for m in methods if m not in out]
        if missing:
            raise DataError(
                f"Requested method(s) with no edge list in {lists_dir}: "
                f"{', '.join(missing)}"
            )
    return out


def load_prior_network(path, genes: List[str], symmetric: bool = False) -> np.ndarray:
    """
    Load a known/curated GRN (RegulonDB, literature, etc.) as a score matrix,
    so it can be stacked alongside inferred algorithms.

    WARNING: a curated network derived from the same source as your evaluation
    ground truth is not a peer of an inference algorithm - using it as an input
    leaks the answer. Keep priors and ground truth from separate sources, or
    hold out the overlap. The pipeline tags these with source_type="prior" so
    the distinction stays visible in results.
    """
    df = pd.read_csv(path, header=None, names=["source", "target", "score"])
    index = {g: i for i, g in enumerate(genes)}
    df = df[df["source"].isin(index) & df["target"].isin(index)]
    if df.empty:
        raise DataError(f"No edges in {path} match the expression matrix genes.")

    n = len(genes)
    mat = np.zeros((n, n), dtype=np.float32)
    mat[df["source"].map(index), df["target"].map(index)] = \
        df["score"].to_numpy(dtype=np.float32)
    if symmetric:
        mat = np.maximum(mat, mat.T)
    return mat
