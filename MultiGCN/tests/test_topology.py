import numpy as np

from multigcn.topology.threshold import density, top_x_percent_edges


def test_top_x_respects_density():
    rng = np.random.default_rng(0)
    m = rng.random((100, 100))
    adj = top_x_percent_edges(m, top_x=0.10)
    assert abs(density(adj) - 0.10) < 0.01


def test_zero_scores_never_become_edges():
    m = np.zeros((50, 50))
    m[0, 1] = 0.9
    adj = top_x_percent_edges(m, top_x=0.50)
    assert adj.sum() == 1


def test_rank_based_is_scale_invariant():
    """The whole point: same ranking, different scale, same topology."""
    rng = np.random.default_rng(2)
    m = rng.random((60, 60))
    a = top_x_percent_edges(m, 0.1)
    b = top_x_percent_edges(m * 1000, 0.1)
    assert np.array_equal(a, b)
