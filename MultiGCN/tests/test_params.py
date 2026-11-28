import pytest

from multigcn import MultiGCN
from multigcn.params import ParamError, split_prefixed
from multigcn.registry import ALGORITHMS, DataRequirement


def test_registry_is_populated_and_unique():
    """Count-free: a hardcoded total breaks every time an algorithm is added."""
    assert len(ALGORITHMS) >= 20
    assert len({s.name for s in ALGORITHMS.values()}) == len(ALGORITHMS)


def test_every_param_prefix_is_unambiguous():
    """No prefix may be a prefix of another, or routing silently misassigns."""
    prefixes = sorted({s.param_prefix for s in ALGORITHMS.values() if s.param_prefix})
    for a in prefixes:
        for b in prefixes:
            if a != b:
                assert not b.startswith(a + "_"), f"{b} shadows {a}"


def test_longest_prefix_wins():
    """dyngenie3_ntrees must not be parsed as the genie3 namespace."""
    assert split_prefixed("dyngenie3_ntrees", ["genie3", "dyngenie3"]) \
        == ("dyngenie3", "ntrees")


def test_params_routed_and_defaults_filled():
    m = MultiGCN(methods=["GENIE3_ET"], genie3_ntrees=500)
    p = m.params_for("GENIE3_ET")
    assert p["ntrees"] == 500 and p["seed"] == 777


def test_genie3_variants_share_namespace():
    m = MultiGCN(methods=["GENIE3_RF", "GRNBOOST2"], genie3_ntrees=250)
    assert m.params_for("GENIE3_RF")["ntrees"] == 250
    assert m.params_for("GRNBOOST2")["ntrees"] == 250


def test_config_kwargs_still_work():
    m = MultiGCN(methods=["GENIE3_ET"], top_x=0.15, epochs=100)
    assert m.config.top_x == 0.15 and m.config.epochs == 100


def test_unknown_param_fails():
    with pytest.raises(ParamError, match="Unknown parameter"):
        MultiGCN(methods=["GENIE3_ET"], genie3_nonsense=1)


def test_bad_choice_fails():
    with pytest.raises(ValueError, match="must be one of"):
        MultiGCN(methods=["BNLEARN"], allow_slow=True, bnlearn_algorithm="not_an_algorithm")


def test_out_of_range_fails():
    with pytest.raises(ValueError, match=">= 2"):
        MultiGCN(methods=["BNLEARN"], allow_slow=True, bnlearn_R=1)


def test_unprefixed_unknown_option_is_helpful():
    with pytest.raises(TypeError, match="prefixed with their package"):
        MultiGCN(methods=["GENIE3_ET"], ntrees=500)


def test_time_series_algorithms_declared():
    assert ALGORITHMS["DYNGENIE3"].requires is DataRequirement.TIME_SERIES
    assert ALGORITHMS["NETWORKBMA"].requires is DataRequirement.TIME_SERIES
    assert ALGORITHMS["DYNUGENE"].requires is DataRequirement.EITHER


def test_time_series_required_raises_before_running(tmp_path):
    m = MultiGCN(methods=["DYNGENIE3"], verbose=False)
    with pytest.raises(ValueError, match="require time-series"):
        m.infer(tmp_path / "anything.csv")


def test_bnlearn_uses_bootstrap_for_continuous_output():
    """A single DAG fit gives binary edges, useless for top-X% ranking."""
    assert "R" in ALGORITHMS["BNLEARN"].params
    assert "boot.strength" in ALGORITHMS["BNLEARN"].notes
