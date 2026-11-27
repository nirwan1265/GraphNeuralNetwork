"""The joint multi-relational model: output contract and cross-algorithm sharing."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from multigcn.training.train import train_joint  # noqa: E402


def _views(seed=0, n=24, keep=0.4, n_views=3):
    """Two 12-gene modules; each 'algorithm' sees a random 40% of the
    within-module pairs plus a few random cross-module pairs."""
    rng = np.random.default_rng(seed)
    module = np.repeat([0, 1], n // 2)
    same = (module[:, None] == module[None, :]) & ~np.eye(n, dtype=bool)
    views = {}
    for v in range(n_views):
        A = np.triu(same & (rng.random((n, n)) < keep), 1).astype(np.float32)
        noise = np.triu(rng.random((n, n)) < 0.02, 1)
        A = np.maximum(A, noise); views[f"alg{v}"] = A + A.T
    return same, views


def test_output_contract():
    _, views = _views()
    out = train_joint(views, epochs=20, seed=0)
    assert set(out) == set(views)
    for m in out.values():
        assert m.shape == (24, 24) and m.dtype == np.float32
        assert m.min() >= 0 and m.max() <= 1 and np.allclose(m, m.T, atol=1e-5)


def test_seeded_runs_are_reproducible():
    _, views = _views()
    a = train_joint(views, epochs=20, seed=3)
    b = train_joint(views, epochs=20, seed=3)
    assert all(np.allclose(a[k], b[k]) for k in a)


def test_recovers_pairs_no_single_algorithm_saw():
    """Within-module pairs missing from EVERY view should still outscore
    cross-module pairs - only possible by pooling structure across views."""
    same, views = _views()
    seen = np.any([v > 0 for v in views.values()], axis=0)
    out = train_joint(views, epochs=150, seed=0)
    fused = np.mean(list(out.values()), axis=0)
    iu = np.triu_indices(24, 1)
    hidden = same[iu] & ~seen[iu]
    cross = ~same[iu] & ~seen[iu]
    assert hidden.sum() > 0
    assert fused[iu][hidden].mean() > fused[iu][cross].mean() + 0.2


def test_pipeline_runs_joint_end_to_end(lists_dir, expression):
    from multigcn import MultiGCN
    m = MultiGCN(methods=["CLR", "ARACNE"], epochs=10, iterations=1, verbose=False)
    m.load(lists_dir, expression_file=expression)
    out = m.fit_predict()
    assert out.shape == (20, 20) and np.allclose(out, out.T)
    assert set(m.reconstructions) == {"CLR", "ARACNE"}
    assert abs(sum(m.weights.values()) - 1) < 1e-9
