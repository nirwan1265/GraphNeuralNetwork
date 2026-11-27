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


# --------------------------------------------------------------------------
# Agreement weighting and the input blend
# --------------------------------------------------------------------------
from multigcn.fusion.combine import agreement_weights, blend_with_inputs, rank_average


def _algorithms(seed=0, n=30):
    """Four noisy views of one true score matrix, plus one pure-noise algorithm."""
    rng = np.random.default_rng(seed)
    truth = rng.random((n, n)); truth = truth + truth.T
    mats = {f"good{k}": truth + 0.5 * rng.random((n, n)) for k in range(4)}
    mats["noise"] = rng.random((n, n))
    return truth, mats


def test_agreement_weights_downweight_the_outlier():
    _, mats = _algorithms()
    w = agreement_weights(mats, power=2.0)
    assert abs(sum(w.values()) - 1) < 1e-9
    assert w["noise"] < min(w[a] for a in w if a != "noise") / 5


def test_agreement_power_zero_is_flat():
    _, mats = _algorithms()
    w = agreement_weights(mats, power=0.0)
    assert np.allclose(list(w.values()), 1 / len(mats))


def test_weighted_fuse_matches_manual_average():
    a, b = np.random.rand(6, 6), np.random.rand(6, 6)
    r = {"A": {"d": [a]}, "B": {"d": [b]}}
    out = fuse(r, {"A": 3.0, "B": 1.0})
    assert np.allclose(out, 0.75 * minmax(a) + 0.25 * minmax(b), atol=1e-6)
    assert np.allclose(fuse(r), fuse(r, {"A": 1, "B": 1}))


def test_blend_zero_share_is_identity_and_output_symmetric():
    _, mats = _algorithms()
    fused = np.random.rand(30, 30)
    assert blend_with_inputs(fused, mats, share=0.0) is fused
    out = blend_with_inputs(fused, mats, share=0.5)
    assert np.allclose(out, out.T) and out.min() >= 0 and out.max() <= 1


def test_rank_average_tracks_the_truth():
    truth, mats = _algorithms()
    iu = np.triu_indices(30, 1)
    ra = rank_average(mats, agreement_weights(mats))
    from scipy.stats import spearmanr
    assert spearmanr(ra[iu], truth[iu]).statistic > 0.8
