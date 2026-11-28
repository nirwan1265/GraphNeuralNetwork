"""
Native Python GRN inference algorithms.

Pure numpy / scipy / scikit-learn implementations, so these run in the same
interpreter as the rest of MultiGCN - no R, Julia, MATLAB or Docker.

    from multigcn.inference.native import infer
    scores = infer("PIDC", expression="expr.csv")          # genes x genes DataFrame
    scores = infer("SCODE", time_series="timeseries.csv")

PIDC, PUC, PLSNET and NONLINEARODES are reimplementations of algorithms that
previously ran only as GENECI containers (Julia / MATLAB / pinned Python 3.6);
each module documents the reference it follows.
"""
from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Optional

import pandas as pd

from ._io import (read_expression, read_knockouts, read_regulators,
                  read_time_series, write_edge_list)
from .correlation import pearson, wgcna
from .dynamics import scode, sincerities
from .graphical import glasso
from .information import pidc, puc
from .perturbation import ko_zscore, portia
from .regression import elasticnet, plsnet
from .trees import nonlinearodes


class NativeAlgorithm(NamedTuple):
    func: Callable
    time_series: bool       # True -> called as func(X, times, ...) on time-series input
    knockouts: str = "no"   # "no", "optional" or "required": passed as X_ko, ko_index


NATIVE: Dict[str, NativeAlgorithm] = {
    "PEARSON": NativeAlgorithm(pearson, False),
    "WGCNA": NativeAlgorithm(wgcna, False),
    "GLASSO": NativeAlgorithm(glasso, False),
    "ELASTICNET": NativeAlgorithm(elasticnet, False),
    "PLSNET": NativeAlgorithm(plsnet, False),
    "NONLINEARODES": NativeAlgorithm(nonlinearodes, False),
    "PIDC": NativeAlgorithm(pidc, False),
    "PUC": NativeAlgorithm(puc, False),
    "SINCERITIES": NativeAlgorithm(sincerities, True),
    "SCODE": NativeAlgorithm(scode, True),
    "PORTIA": NativeAlgorithm(portia, False, "optional"),
    "KO_ZSCORE": NativeAlgorithm(ko_zscore, False, "required"),
}


# Algorithms that can restrict their predictors to a regulator list.
REGULATOR_AWARE = {"ELASTICNET", "PORTIA"}


def infer(name: str, expression=None, time_series=None, knockouts=None,
          regulators=None, **params) -> pd.DataFrame:
    """
    Run one native algorithm and return its genes x genes score matrix
    (row = regulator, column = target).

    knockouts: single-gene knockout experiments (see _io.read_knockouts);
    used by PORTIA (optional) and KO_ZSCORE (required), ignored otherwise.
    """
    if name not in NATIVE:
        raise ValueError(f"Unknown native algorithm {name!r}. "
                         f"Available: {', '.join(NATIVE)}")
    algo = NATIVE[name]
    if algo.time_series:
        if time_series is None:
            raise ValueError(f"{name} requires time-series input.")
        X, times, _, genes = read_time_series(time_series)
        S = algo.func(X, times, **params)
    else:
        if expression is None:
            raise ValueError(f"{name} requires an expression file.")
        X, genes = read_expression(expression)
        if regulators is not None and name in REGULATOR_AWARE:
            params = {**params, "regulators": read_regulators(regulators, genes)}
        if algo.knockouts != "no" and knockouts is not None:
            X_ko, ko_index = read_knockouts(knockouts, genes)
            params = {**params, "X_ko": X_ko, "ko_index": ko_index}
        elif algo.knockouts == "required":
            raise ValueError(f"{name} requires knockout experiments: pass knockouts=.")
        S = algo.func(X, **params)
    return pd.DataFrame(S, index=genes, columns=genes)


def run_to_file(name: str, out_dir, expression=None, time_series=None,
                knockouts=None, regulators=None, **params):
    """Run one algorithm and write GRN_<name>.csv in the package's edge-list format."""
    S = infer(name, expression=expression, time_series=time_series,
              knockouts=knockouts, regulators=regulators, **params)
    return write_edge_list(S.to_numpy(), list(S.index), out_dir, name)


__all__ = ["NATIVE", "infer", "run_to_file"]
