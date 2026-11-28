"""
Tests for the native Python algorithms.

Each steady-state algorithm must recover a planted regulatory link from
synthetic data well above chance; each time-series algorithm must recover a
planted lagged link. Plus the output contract and the registry wiring.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from multigcn.inference.native import NATIVE, infer
from multigcn.inference.native.information import (bayesian_blocks_edges,
                                                   discretize, puc_scores)
from multigcn.inference.native.specs import NATIVE_NAMES
from multigcn.registry import ALGORITHMS, DataRequirement

SCRIPT = (Path(__file__).resolve().parents[1]
          / "multigcn/inference/scripts/python/native.py")

FAST = {
    "PLSNET": {"n_iter": 20, "n_jobs": 1},
    "NONLINEARODES": {"n_estimators": 30, "n_jobs": 1},
    "ELASTICNET": {"n_jobs": 1},
    "SINCERITIES": {"n_lambdas": 5, "n_jobs": 1},
    "SCODE": {"n_runs": 3, "n_iter": 30},
}


@pytest.fixture(scope="module")
def steady(tmp_path_factory):
    """12 genes, 300 samples; G0 -> G1 and G0 -> G2 strongly, rest noise."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12))
    X[:, 1] = 0.9 * X[:, 0] + 0.3 * rng.normal(size=300)
    X[:, 2] = np.tanh(2 * X[:, 0]) + 0.3 * rng.normal(size=300)
    genes = [f"G{i}" for i in range(12)]
    path = tmp_path_factory.mktemp("steady") / "expr.csv"
    pd.DataFrame(X.T, index=genes,
                 columns=[f"S{k}" for k in range(300)]).to_csv(path)
    return path


@pytest.fixture(scope="module")
def timeseries(tmp_path_factory):
    """10 experiments x 21 time points; G0 drives G1 with a one-step lag."""
    rng = np.random.default_rng(1)
    rows = []
    for e in range(10):
        x = rng.normal(size=10)
        level = rng.uniform(-2, 2)
        for k, t in enumerate(range(0, 1050, 50)):
            x_new = 0.6 * x + 0.3 * rng.normal(size=10)
            x_new[0] = level * np.exp(-k / 5) + 0.1 * rng.normal()
            x_new[1] = 0.8 * x[0] + 0.1 * rng.normal()
            x = x_new
            rows.append([t] + list(x))
    df = pd.DataFrame(rows, columns=["time"] + [f"G{i}" for i in range(10)])
    path = tmp_path_factory.mktemp("ts") / "ts.csv"
    df.to_csv(path, index=False)
    return path


def _rank_of(S, i, j):
    """Rank (1 = best) of edge (i, j) among all off-diagonal pairs, undirected."""
    M = np.maximum(S, S.T)
    iu = np.triu_indices(M.shape[0], 1)
    vals = M[iu]
    target = M[min(i, j), max(i, j)]
    return int((vals > target).sum()) + 1


def test_all_native_registered():
    assert set(NATIVE) == set(NATIVE_NAMES)
    for name in NATIVE_NAMES:
        spec = ALGORITHMS[name]
        assert spec.script == "python/native.py"
        assert spec.extra_args == [name]
        assert spec.params and spec.param_prefix


def test_time_series_flags_agree():
    for name, algo in NATIVE.items():
        needs_ts = ALGORITHMS[name].requires is DataRequirement.TIME_SERIES
        assert needs_ts == algo.time_series, name
        needs_ko = ALGORITHMS[name].requires is DataRequirement.KNOCKOUT
        assert needs_ko == (algo.knockouts == "required"), name


# PORTIA is exempt: its gene-bias correction divides each score by the gene's
# mean partial correlation, which in a 12-gene toy with one 2-target hub
# demotes the hub's own edges (rank 19/66). The wrapper reproduces PORTIA's
# published DREAM4 predictions exactly (AUPR 0.461 with knockouts, 0.062
# without), and test_portia_runs_with_and_without_knockouts covers it here.
@pytest.mark.parametrize("name", [n for n, a in NATIVE.items()
                                  if not a.time_series and a.knockouts == "no"])
def test_steady_state_recovers_planted_edges(steady, name):
    S = infer(name, expression=steady, **FAST.get(name, {})).to_numpy()
    assert S.shape == (12, 12)
    assert np.isfinite(S).all() and (S >= 0).all()
    assert np.allclose(np.diag(S), 0)
    # 66 undirected pairs; the two planted edges must both be in the top 5.
    assert _rank_of(S, 0, 1) <= 5, name
    assert _rank_of(S, 0, 2) <= 5, name


def test_sincerities_recovers_lagged_edge(timeseries):
    S = infer("SINCERITIES", time_series=timeseries,
              **FAST["SINCERITIES"]).to_numpy()
    assert S.shape == (10, 10)
    assert np.isfinite(S).all() and (S >= 0).all()
    assert _rank_of(S, 0, 1) <= 10            # top ~20% of 45 pairs


def test_scode_recovers_its_own_generative_model():
    """
    SCODE assumes ONE shared trajectory x(t) = W exp(b t). Perturbation
    experiments with different directions violate that, so the meaningful
    implementation test is: simulate exactly SCODE's model and check the
    recovered A matches the true W diag(b) W^+.
    """
    from scipy.stats import spearmanr
    from multigcn.inference.native.dynamics import scode
    rng = np.random.default_rng(6)
    G = 15
    W = rng.normal(size=(G, 4))
    b = np.array([-6.0, -2.5, -0.8, 1.2])
    true_S = np.abs(W @ np.diag(b) @ np.linalg.pinv(W)).T
    t = np.tile(np.linspace(0, 1, 21), 10)
    X = (W @ np.exp(np.outer(b, t))).T + 0.01 * rng.normal(size=(len(t), G))
    S = scode(X, t, n_runs=5, n_iter=200)
    off = ~np.eye(G, dtype=bool)
    assert spearmanr(S[off], true_S[off]).statistic > 0.8


def test_time_series_algorithm_refuses_steady_input(steady):
    with pytest.raises(ValueError, match="time-series"):
        infer("SCODE", expression=steady)


def test_bayesian_blocks_spans_the_gap():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0, 0.1, 200), rng.normal(5, 0.1, 200)])
    e = bayesian_blocks_edges(x)
    # one block must bridge the empty region between the clusters
    assert any(lo < 1 and hi > 4 for lo, hi in zip(e[:-1], e[1:]))


def test_bayesian_blocks_matches_astropy():
    astropy = pytest.importorskip("astropy.stats")
    rng = np.random.default_rng(0)
    for x in [rng.normal(size=300), rng.exponential(size=250),
              np.tanh(2 * rng.normal(size=300)) + 0.3 * rng.normal(size=300)]:
        ref = astropy.bayesian_blocks(x, fitness="events", p0=0.05)
        assert np.allclose(bayesian_blocks_edges(x), ref)


def test_puc_is_symmetric_and_bounded():
    rng = np.random.default_rng(4)
    codes = discretize(rng.normal(size=(200, 6)), "uniform_count", 4)
    S = puc_scores(codes)
    assert np.allclose(S, S.T)
    assert (S >= -1e-9).all()


def test_dispatcher_writes_edge_list(steady, tmp_path):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(steady), str(tmp_path), "PEARSON",
         json.dumps({"method": "spearman"})],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    df = pd.read_csv(tmp_path / "GRN_PEARSON.csv", header=None)
    assert df.shape[1] == 3
    assert (df[0] != df[1]).all()
    assert df[2].between(0, 1).all()
    assert df[2].nunique() > 2          # continuous, not binarised


# --------------------------------------------------------------------------
# Knockout experiments: KO_ZSCORE and PORTIA
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def knockout_data(tmp_path_factory):
    """12 genes; G0 -> G1 and G0 -> G2. Wild-type samples + one knockout per gene
    (DREAM layout: row k = knockout of gene k). Knocking out G0 drops G1 and G2."""
    rng = np.random.default_rng(5)
    genes = [f"G{i}" for i in range(12)]

    def simulate(ko=None, n=1):
        X = rng.uniform(0.4, 0.6, size=(n, 12))
        if ko is not None:
            X[:, ko] = 0.0
        X[:, 1] = 0.8 * X[:, 0] + 0.05 * rng.normal(size=n) + 0.1
        X[:, 2] = 0.7 * X[:, 0] + 0.05 * rng.normal(size=n) + 0.1
        return X

    d = tmp_path_factory.mktemp("ko")
    expr = simulate(n=60)
    pd.DataFrame(expr.T, index=genes,
                 columns=[f"S{k}" for k in range(60)]).to_csv(d / "expr.csv")
    ko = np.vstack([simulate(ko=k) for k in range(12)])
    pd.DataFrame(ko, columns=genes).to_csv(d / "knockouts.tsv", sep="\t", index=False)
    return d / "expr.csv", d / "knockouts.tsv"


def test_ko_zscore_finds_the_regulator(knockout_data):
    expr, ko = knockout_data
    S = infer("KO_ZSCORE", expression=expr, knockouts=ko).to_numpy()
    assert S.shape == (12, 12) and np.allclose(np.diag(S), 0)
    # directed: knocking out G0 moves G1/G2, not the reverse
    assert S[0, 1] > S[1, 0] and S[0, 2] > S[2, 0]
    top = np.argsort(S.ravel())[::-1][:2]
    assert set(map(tuple, np.column_stack(np.unravel_index(top, S.shape)))) == {(0, 1), (0, 2)}


def test_ko_zscore_requires_knockouts(knockout_data):
    expr, _ = knockout_data
    with pytest.raises(ValueError, match="knockout"):
        infer("KO_ZSCORE", expression=expr)


def test_knockouts_with_explicit_column(knockout_data, tmp_path):
    expr, ko = knockout_data
    df = pd.read_csv(ko, sep="\t").iloc[::-1]               # shuffled row order
    df.insert(0, "knockout", [f"G{i}" for i in range(11, -1, -1)])
    df.to_csv(tmp_path / "ko.csv", index=False)
    a = infer("KO_ZSCORE", expression=expr, knockouts=ko).to_numpy()
    b = infer("KO_ZSCORE", expression=expr, knockouts=tmp_path / "ko.csv").to_numpy()
    assert np.allclose(a, b)


def test_portia_runs_with_and_without_knockouts(knockout_data):
    pytest.importorskip("portia")
    expr, ko = knockout_data
    with_ko = infer("PORTIA", expression=expr, knockouts=ko).to_numpy()
    no_ko = infer("PORTIA", expression=expr).to_numpy()
    for S in (with_ko, no_ko):
        assert S.shape == (12, 12) and np.isfinite(S).all() and np.allclose(np.diag(S), 0)
    assert _rank_of(with_ko, 0, 1) <= 3 and _rank_of(with_ko, 0, 2) <= 3


def test_dispatcher_passes_knockouts(knockout_data, tmp_path):
    expr, ko = knockout_data
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(expr), str(tmp_path), "KO_ZSCORE",
         json.dumps({"knockouts": str(ko)})], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "GRN_KO_ZSCORE.csv").exists()


# --------------------------------------------------------------------------
# Candidate regulators (TF list)
# --------------------------------------------------------------------------

def test_elasticnet_scores_only_listed_regulators(steady, tmp_path):
    tf = tmp_path / "tfs.txt"; tf.write_text("G0\nG5\nNOT_A_GENE\n")
    S = infer("ELASTICNET", expression=steady, regulators=tf, n_jobs=1).to_numpy()
    rows = np.flatnonzero(S.sum(axis=1) > 0)
    assert set(rows) <= {0, 5}
    assert _rank_of(S, 0, 1) <= 2 and _rank_of(S, 0, 2) <= 2


def test_genie3_script_restricts_to_regulators(steady, tmp_path):
    script = SCRIPT.parent / "GENIE3.py"
    tf = tmp_path / "tfs.txt"; tf.write_text("G0\nG5\n")
    out = subprocess.run(
        [sys.executable, str(script), str(steady), str(tmp_path), "ET",
         json.dumps({"ntrees": 20, "n_jobs": 1, "regulators": str(tf)})],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    df = pd.read_csv(tmp_path / "GRN_GENIE3_ET.csv", header=None)
    assert set(df[0]) <= {"G0", "G5"}
    # Importances sum to 1 per target, so with two TFs each is the sole
    # predictor of the other; compare them on the non-TF targets.
    S = df.pivot(index=0, columns=1, values=2).fillna(0)
    for target in ("G1", "G2"):
        assert S.loc["G0", target] > 5 * S.loc["G5", target]
