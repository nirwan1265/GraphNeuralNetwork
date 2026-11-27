"""Evaluation helpers."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def _flat(pred: np.ndarray, truth: np.ndarray, exclude_diagonal: bool = True,
          undirected: bool = True):
    """
    Flatten a prediction/truth pair for scoring.

    undirected=True (the validated default) scores the UPPER TRIANGLE only:
    each gene pair counted once, direction ignored. This is not cosmetic - it
    changes both the number of samples and the class balance, so AUPR computed
    over the full n^2-n ordered pairs is NOT comparable to AUPR over the
    n(n-1)/2 unordered ones. Mixing the two silently makes results look worse
    or better for no methodological reason.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = (np.asarray(truth) > 0).astype(np.int8)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs truth {truth.shape}")

    if undirected:
        # A directed gold standard has to be collapsed too, or a true edge
        # recorded only as j->i would be counted as a negative at (i, j).
        triu = np.triu_indices(pred.shape[0], k=1)
        truth_sym = np.maximum(truth, truth.T)
        return pred[triu], truth_sym[triu]

    if exclude_diagonal:
        mask = ~np.eye(pred.shape[0], dtype=bool)
        return pred[mask], truth[mask]
    return pred.ravel(), truth.ravel()


def ranking_metrics(scores: np.ndarray, truth: np.ndarray,
                    undirected: bool = True) -> Dict[str, float]:
    """
    AUROC and AUPR on the CONTINUOUS score, before any cutoff.

    Never compute these on a post-cutoff binary network. They measure ranking
    quality across every possible cutoff; computed after committing to one they
    will always look worse, and that is a property of the metric, not a failure
    of the method.

    AUPR is the priority metric here given the class imbalance (DREAM4 net3 is
    1.94% true edges).
    """
    p, t = _flat(scores, truth, undirected=undirected)
    return {"auroc": float(roc_auc_score(t, p)),
            "aupr": float(average_precision_score(t, p))}


def binary_metrics(predicted: np.ndarray, truth: np.ndarray,
                   undirected: bool = True) -> Dict[str, float]:
    """Precision/recall/F1 for the actual final network a cutoff produced."""
    p, t = _flat(predicted, truth, undirected=undirected)
    p = (p > 0).astype(np.int8)
    return {
        "precision": float(precision_score(t, p, zero_division=0)),
        "recall": float(recall_score(t, p, zero_division=0)),
        "f1": float(f1_score(t, p, zero_division=0)),
        "predicted_edges": int(p.sum()),
        "true_edges": int(t.sum()),
    }


def load_gold_standard(path, genes) -> np.ndarray:
    """
    Load a gold standard, accepting either layout DREAM ships:

      matrix     genes x genes, first column and header row are gene ids
                 (this is what data/Database.DREAM4/GS/*.csv uses)
      edge list  headerless source,target,label - label 0/1

    Detected from the file rather than configured, because getting it wrong is
    silent: an edge list parsed as a matrix yields an all-zero truth and an
    AUPR of 0 that looks like a modelling failure.
    """
    import pandas as pd

    path = str(path)
    index = {g: i for i, g in enumerate(genes)}
    mat = np.zeros((len(genes), len(genes)), dtype=np.int8)

    head = pd.read_csv(path, sep=None, engine="python", nrows=1, header=None)

    # A matrix has one column per gene (plus the label column); an edge list
    # has exactly three.
    if head.shape[1] > 3:
        df = pd.read_csv(path, sep=None, engine="python", index_col=0)
        missing = [g for g in genes if g not in df.index or g not in df.columns]
        if missing:
            raise ValueError(
                f"Gold-standard matrix at {path} is missing {len(missing)} "
                f"gene(s) present in the expression data, e.g. {missing[:5]}."
            )
        sub = df.loc[genes, genes].to_numpy()
        return (sub > 0).astype(np.int8)

    df = pd.read_csv(path, sep=None, engine="python", header=None,
                     names=["source", "target", "label"])
    df = df[(df["label"] > 0) & df["source"].isin(index) & df["target"].isin(index)]
    if df.empty:
        raise ValueError(
            f"No positive edges in {path} matched the expression genes. "
            "Check that gene names agree between the two files."
        )
    mat[df["source"].map(index), df["target"].map(index)] = 1
    return mat
