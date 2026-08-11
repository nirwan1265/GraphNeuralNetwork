import numpy as np

from multigcn.fusion.combine import continuous_average, fuse, minmax


def test_minmax_range():
    out = minmax(np.array([[2.0, 4.0], [6.0, 8.0]]))
    assert out.min() == 0.0 and out.max() == 1.0


def test_constant_matrix_does_not_nan():
    assert not np.isnan(minmax(np.ones((5, 5)))).any()


def test_continuous_average_preserves_confidence():
    """Averaging must not collapse to a vote count."""
    a = np.array([[0.0, 1.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.4], [1.0, 0.0]])
    out = continuous_average([a, b])
    assert out[0, 1] < out[1, 0]  # weaker agreement ranks lower


def test_fuse_nested_structure():
    r = {"CLR": {"bilinear": [np.random.rand(10, 10)],
                 "distance": [np.random.rand(10, 10)]}}
    assert fuse(r).shape == (10, 10)
