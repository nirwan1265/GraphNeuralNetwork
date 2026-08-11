import numpy as np
import pytest

from multigcn.evaluation.metrics import binary_metrics, ranking_metrics


def test_perfect_ranking():
    truth = np.zeros((10, 10), dtype=int)
    truth[0, 1] = truth[2, 3] = 1
    scores = truth.astype(float) * 0.9 + 0.05
    r = ranking_metrics(scores, truth)
    assert r["auroc"] == pytest.approx(1.0)
    assert r["aupr"] == pytest.approx(1.0)


def test_diagonal_excluded():
    truth = np.eye(10, dtype=int)
    scores = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        ranking_metrics(scores, truth[:5])


def test_binary_metrics_counts():
    truth = np.zeros((10, 10), dtype=int); truth[0, 1] = 1
    pred = np.zeros((10, 10), dtype=int); pred[0, 1] = pred[3, 4] = 1
    m = binary_metrics(pred, truth)
    assert m["predicted_edges"] == 2 and m["true_edges"] == 1
    assert m["recall"] == pytest.approx(1.0)
