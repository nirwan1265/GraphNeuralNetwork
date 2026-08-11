"""Evaluation helpers."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def _flat(pred: np.ndarray, truth: np.ndarray, exclude_diagonal: bool = True):
    pred = np.asarray(pred, dtype=np.float64)
    truth = (np.asarray(truth) > 0).astype(np.int8)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs truth {truth.shape}")
    if exclude_diagonal:
        mask = ~np.eye(pred.shape[0], dtype=bool)
        return pred[mask], truth[mask]
    return pred.ravel(), truth.ravel()


def ranking_metrics(scores: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """
    AUROC and AUPR on the CONTINUOUS score, before any cutoff.

    Never compute these on a post-cutoff binary network. They measure ranking
    quality across every possible cutoff; computed after committing to one they
    will always look worse, and that is a property of the metric, not a failure
    of the method.

    AUPR is the priority metric here given the ~3.5% true-edge class imbalance.
    """
    p, t = _flat(scores, truth)
    return {"auroc": float(roc_auc_score(t, p)),
            "aupr": float(average_precision_score(t, p))}


def binary_metrics(predicted: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Precision/recall/F1 for the actual final network a cutoff produced."""
    p, t = _flat(predicted, truth)
    p = (p > 0).astype(np.int8)
    return {
        "precision": float(precision_score(t, p, zero_division=0)),
        "recall": float(recall_score(t, p, zero_division=0)),
        "f1": float(f1_score(t, p, zero_division=0)),
        "predicted_edges": int(p.sum()),
        "true_edges": int(t.sum()),
    }


def load_gold_standard(path, genes) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(path, sep=None, engine="python", header=None,
                     names=["source", "target", "label"])
    index = {g: i for i, g in enumerate(genes)}
    df = df[(df["label"] > 0) & df["source"].isin(index) & df["target"].isin(index)]
    mat = np.zeros((len(genes), len(genes)), dtype=np.int8)
    mat[df["source"].map(index), df["target"].map(index)] = 1
    return mat
