"""
Registry entries for the native Python algorithms.

Kept in its own module and pulled into registry.py with a single import, so
adding this set touches one line of the shared registry. Four of these
(PIDC, PUC, PLSNET, NONLINEARODES) supersede the old container-only entries:
same algorithm, now runnable in-process.
"""
from __future__ import annotations

from ...params import ParamSpec
from ...registry import (ALGORITHMS, AlgorithmSpec, DataRequirement,
                         Orientation, Runtime)

_SCRIPT = "python/native.py"


def _native(name, family, requires, prefix, params, library, notes):
    """Register (or replace) a native algorithm."""
    ALGORITHMS[name] = AlgorithmSpec(
        name=name,
        runtime=Runtime.PYTHON,
        script=_SCRIPT,
        library=library,
        install="included - native Python",
        orientation=Orientation.SAMPLES_ROWS,
        family=family,
        extra_args=[name],
        requires=requires,
        param_prefix=prefix,
        params=params,
        vendored=False,
        notes=notes,
    )


_SS = DataRequirement.STEADY_STATE
_TS = DataRequirement.TIME_SERIES

_JOBS = ParamSpec("n_jobs", -1, int, help="Parallel workers; -1 = all cores.")
_SEED = ParamSpec("seed", 777, int, help="Random seed.")

# --- co-expression -----------------------------------------------------------
_native("PEARSON", "correlation", _SS, "pearson", {
    "method": ParamSpec("method", "pearson", str, choices=["pearson", "spearman"]),
}, "numpy", "Absolute correlation. The baseline every comparison needs.")

_native("WGCNA", "correlation", _SS, "wgcna", {
    "beta": ParamSpec("beta", 0, int, minimum=0,
                      help="Soft-threshold power; 0 = pick automatically "
                           "(WGCNA pickSoftThreshold rule)."),
    "network_type": ParamSpec("network_type", "unsigned", str,
                              choices=["unsigned", "signed"]),
    "rsquared_cut": ParamSpec("rsquared_cut", 0.85, float, minimum=0.0, maximum=1.0),
}, "numpy (WGCNA adjacency + TOM)",
   "Topological overlap of the soft-thresholded co-expression network.")

# --- conditional dependence --------------------------------------------------
_native("GLASSO", "partial_correlation", _SS, "glasso", {
    "n_alphas": ParamSpec("n_alphas", 12, int, minimum=2),
    "alpha_min_ratio": ParamSpec("alpha_min_ratio", 0.05, float,
                                 minimum=1e-4, maximum=1.0),
    "max_iter": ParamSpec("max_iter", 200, int, minimum=10),
}, "scikit-learn", "Graphical lasso partial correlations integrated over the "
                   "penalty path, so every pair gets a continuous score.")

# --- regression --------------------------------------------------------------
_native("ELASTICNET", "regression", _SS, "elasticnet", {
    "l1_ratio": ParamSpec("l1_ratio", 0.5, float, minimum=0.0, maximum=1.0),
    "n_alphas": ParamSpec("n_alphas", 50, int, minimum=2),
    "eps": ParamSpec("eps", 1e-3, float, minimum=1e-6, maximum=0.5),
    "n_jobs": _JOBS,
}, "scikit-learn", "Per-target elastic net, mean |coefficient| over the path.")

_native("PLSNET", "regression", _SS, "plsnet", {
    "n_components": ParamSpec("n_components", 5, int, minimum=1),
    "n_iter": ParamSpec("n_iter", 1000, int, minimum=1,
                        help="Bootstrap rounds (reference: 1000)."),
    "seed": _SEED, "n_jobs": _JOBS,
}, "numpy (SIMPLS)", "Port of the reference MATLAB: bootstrap SIMPLS VIP + "
                     "variance refinement. Previously container-only.")

_native("NONLINEARODES", "tree_ensemble", _SS, "nonlinearodes", {
    "n_estimators": ParamSpec("n_estimators", 0, int, minimum=0,
                              help="0 = GENECI rule round(10 n / log10 n)."),
    "max_depth": ParamSpec("max_depth", 5, int, minimum=1),
    "learning_rate": ParamSpec("learning_rate", 0.3, float, minimum=1e-4, maximum=1.0),
    "seed": _SEED, "n_jobs": _JOBS,
}, "xgboost if installed, else scikit-learn",
   "Boosted-tree split-count importance, as GENECI ran it (no ODE fitting). "
   "Output orientation corrected relative to GENECI. Previously container-only.")

# --- information decomposition ----------------------------------------------
_PID = {
    "discretizer": ParamSpec("discretizer", "bayesian_blocks", str,
                             choices=["bayesian_blocks", "uniform_count"]),
    "n_bins": ParamSpec("n_bins", 0, int, minimum=0,
                        help="uniform_count only; 0 = N^(1/3)."),
}
_native("PIDC", "information_decomposition", _SS, "pidc", dict(_PID),
        "numpy (Williams-Beer PID)",
        "PUC in per-gene CDF context, NetworkInference.jl defaults. "
        "Previously container-only (Julia).")
_native("PUC", "information_decomposition", _SS, "pidc", dict(_PID),
        "numpy (Williams-Beer PID)",
        "Proportional unique contribution. Shares the pidc_ parameters. "
        "Previously container-only (Julia).")

# --- time series ---------------------------------------------------------------
_native("SINCERITIES", "dynamic_regression", _TS, "sincerities", {
    "normalize_dt": ParamSpec("normalize_dt", True, bool),
    "n_lambdas": ParamSpec("n_lambdas", 20, int, minimum=2),
    "n_jobs": _JOBS,
}, "scipy", "KS distribution shifts between time points + non-negative ridge. "
            "Samples at the same time point across experiments act as the cells.")

_native("SCODE", "ode", _TS, "scode", {
    "n_latent": ParamSpec("n_latent", 4, int, minimum=1),
    "n_iter": ParamSpec("n_iter", 100, int, minimum=1),
    "n_runs": ParamSpec("n_runs", 20, int, minimum=1),
    "b_min": ParamSpec("b_min", -10.0, float),
    "b_max": ParamSpec("b_max", 2.0, float),
    "seed": _SEED,
}, "numpy", "Linear ODE through a latent trajectory, averaged over runs.")


NATIVE_NAMES = ["PEARSON", "WGCNA", "GLASSO", "ELASTICNET", "PLSNET",
                "NONLINEARODES", "PIDC", "PUC", "SINCERITIES", "SCODE"]
