import pytest
from multigcn.registry import ALGORITHMS, Runtime, resolve


GENECI_16 = {
    "ARACNE", "BC3NET", "C3NET", "CLR", "GENIE3_RF", "GRNBOOST2", "GENIE3_ET",
    "MRNET", "MRNETB", "PCIT", "TIGRESS", "KBOOST", "MEOMI", "LEAP",
    "NONLINEARODES", "INFERELATOR",
}
MULTIGCN_ADDED = {
    "DYNGENIE3", "DYNUGENE", "NETWORKBMA", "BNLEARN",
    "PPCOR", "GENENET", "G1DBN", "SCANBMA", "TDARACNE",
}


def test_geneci_sixteen_all_present():
    """The originally benchmarked set must stay intact and reproducible."""
    assert GENECI_16 <= set(ALGORITHMS)


def test_added_algorithms_present():
    assert MULTIGCN_ADDED <= set(ALGORITHMS)


def test_every_added_algorithm_has_a_wrapper_declared():
    for name in MULTIGCN_ADDED:
        spec = ALGORITHMS[name]
        assert spec.script, f"{name} has no wrapper script"
        assert spec.vendored is False, f"{name} should be MultiGCN-owned"


def test_added_algorithms_expose_parameters():
    """The whole point of owning these wrappers is that they are tunable."""
    for name in MULTIGCN_ADDED:
        assert ALGORITHMS[name].params, f"{name} exposes no parameters"
        assert ALGORITHMS[name].param_prefix, f"{name} has no param prefix"


def test_resolve_all():
    assert len(resolve(all_methods=True)) == len(ALGORITHMS)


def test_unknown_method_fails_loudly():
    with pytest.raises(ValueError, match="Unknown method"):
        resolve(["NOT_A_REAL_ALGORITHM"])


def test_empty_selection_fails():
    with pytest.raises(ValueError, match="No methods selected"):
        resolve()


def test_former_container_algorithms_now_native():
    """PIDC, PUC, PLSNET, NONLINEARODES were reimplemented in Python."""
    for name in ["NONLINEARODES", "PIDC", "PUC", "PLSNET"]:
        spec = ALGORITHMS[name]
        assert spec.runtime == Runtime.PYTHON, name
        assert spec.script == "python/native.py", name
        assert spec.extra_args == [name], name


def test_genie3_variants_share_one_script():
    scripts = {ALGORITHMS[n].script for n in ["GENIE3_RF", "GENIE3_ET", "GRNBOOST2"]}
    assert len(scripts) == 1
    assert ALGORITHMS["GENIE3_ET"].extra_args == ["ET"]


DOCKER_ONLY = {
    "JUMP3", "NARROMI", "GRNVBEM", "CMI2NI",
    "PCACMI", "LOCPCACMI", "RSNET",
}


def test_docker_algorithms_have_images_and_no_script():
    """Container-only tools must declare an image and must not claim a wrapper."""
    for name in DOCKER_ONLY:
        spec = ALGORITHMS[name]
        assert spec.runtime is Runtime.DOCKER, name
        assert spec.docker_image, f"{name} has no image"
        assert spec.script is None, f"{name} should have no wrapper script"


def test_all_geneci_images_use_one_tag():
    """A stale tag silently pulls a different algorithm version."""
    tags = {s.docker_image.rsplit(":", 1)[1]
            for s in ALGORITHMS.values()
            if s.docker_image and "geneci" in s.docker_image}
    assert tags == {"4.0.0"}, f"mixed GENECI image tags: {tags}"


def test_swing_is_the_only_granger_method():
    from multigcn.registry import by_family
    assert by_family()["granger_causality"] == ["SWING"]
