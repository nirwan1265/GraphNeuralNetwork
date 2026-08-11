"""
Algorithm registry.

Single source of truth for every inference method MultiGCN can run: which
runtime it needs, which vendored wrapper script implements it, what the
underlying dependency is, and how the input matrix must be oriented.

Adding a new algorithm = adding one AlgorithmSpec entry here. Nothing else
in the package needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .params import ParamSpec


class Runtime(str, Enum):
    """How an algorithm has to be executed."""
    R = "r"              # Rscript, local R installation
    PYTHON = "python"    # same interpreter as MultiGCN
    PYTHON_ISOLATED = "python_isolated"  # incompatible deps -> own venv or container
    DOCKER = "docker"    # container only
    NATIVE = "native"    # implemented inside MultiGCN itself (e.g. your GCN, priors)


class DataRequirement(str, Enum):
    """
    What kind of expression data an algorithm needs.

    This is not a detail. dynGENIE3 and networkBMA are *dynamical* methods:
    they model expression change between consecutive time points and cannot
    run on steady-state data at all. Selecting them with a steady-state-only
    dataset is a hard error, not a degraded result.
    """
    STEADY_STATE = "steady_state"
    TIME_SERIES = "time_series"
    EITHER = "either"


class Orientation(str, Enum):
    """
    Expected orientation of the expression matrix as handed to the underlying
    library. The canonical on-disk format is GENES x SAMPLES (genes as rows,
    first column = gene id, header row = sample ids).

    GENES_ROWS      -> library consumes the file as-is
    SAMPLES_ROWS    -> wrapper transposes before calling the library

    This is not cosmetic. Passing a transposed matrix produces a network that
    looks fine and means nothing.
    """
    GENES_ROWS = "genes_rows"
    SAMPLES_ROWS = "samples_rows"


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    runtime: Runtime
    script: Optional[str]           # path relative to multigcn/inference/scripts/
    library: str                    # underlying implementation, for provenance
    install: str                    # how to obtain `library`
    orientation: Orientation
    family: str                     # methodological family, for ablation grouping
    extra_args: List[str] = field(default_factory=list)
    docker_image: Optional[str] = None
    notes: str = ""
    requires: DataRequirement = DataRequirement.STEADY_STATE
    param_prefix: Optional[str] = None
    params: Dict[str, ParamSpec] = field(default_factory=dict)
    vendored: bool = True   # False = wrapper shipped by MultiGCN, not GENECI

    @property
    def tunable(self) -> bool:
        return bool(self.params)

    @property
    def output_filename(self) -> str:
        return f"GRN_{self.name}.csv"


# --------------------------------------------------------------------------
# The 16 techniques validated in the DREAM4 benchmark.
# --------------------------------------------------------------------------

_R = Runtime.R
_PY = Runtime.PYTHON
_ISO = Runtime.PYTHON_ISOLATED
_GR = Orientation.GENES_ROWS
_SR = Orientation.SAMPLES_ROWS

ALGORITHMS: Dict[str, AlgorithmSpec] = {}


def register(spec: AlgorithmSpec) -> AlgorithmSpec:
    if spec.name in ALGORITHMS:
        raise ValueError(f"Algorithm {spec.name!r} is already registered")
    ALGORITHMS[spec.name] = spec
    return spec


# --- Mutual information / correlation, all via Bioconductor `minet` --------
for _name, _method in [
    ("ARACNE", "aracne"),
    ("CLR", "clr"),
    ("MRNET", "mrnet"),
    ("MRNETB", "mrnetb"),
]:
    register(AlgorithmSpec(
        name=_name,
        runtime=_R,
        script=f"r/{_name}.R",
        library="minet",
        install="BiocManager::install('minet')",
        orientation=_SR,
        family="mutual_information",
    ))

register(AlgorithmSpec(
    name="BC3NET", runtime=_R, script="r/BC3NET.R",
    library="bc3net", install="install.packages('bc3net')",
    orientation=_GR, family="mutual_information",
    notes="Runs 11 bootstrap replicates internally and takes the median. Slow.",
))

register(AlgorithmSpec(
    name="C3NET", runtime=_R, script="r/C3NET.R",
    library="c3net", install="install.packages('c3net')",
    orientation=_GR, family="mutual_information",
))

register(AlgorithmSpec(
    name="PCIT", runtime=_R, script="r/PCIT.R",
    library="CeTF", install="BiocManager::install('CeTF')",
    orientation=_GR, family="correlation",
    notes="Uses CeTF::PCIT, not a package named 'pcit'. Takes |correlation|; "
          "regulation direction is discarded upstream.",
))

# --- Tree ensemble / boosting ---------------------------------------------
_GENIE3_PARAMS = {
    "ntrees": ParamSpec("ntrees", 1000, int, minimum=1,
                        help="Trees per ensemble (arboreto default 1000)."),
    "max_features": ParamSpec("max_features", "sqrt", str,
                              choices=["sqrt", "log2", "all"],
                              help="Candidate regulators considered per split."),
    "n_jobs": ParamSpec("n_jobs", -1, int,
                        help="Parallel workers; -1 uses all cores."),
    "seed": ParamSpec("seed", 777, int, help="Random seed."),
}

for _name, _variant in [("GENIE3_RF", "RF"), ("GENIE3_ET", "ET"), ("GRNBOOST2", "GBM")]:
    register(AlgorithmSpec(
        name=_name, runtime=_PY, script="python/GENIE3.py",
        library="scikit-learn", install="included in core dependencies",
        orientation=_SR, family="tree_ensemble",
        extra_args=[_variant],
        param_prefix="genie3", params=dict(_GENIE3_PARAMS), vendored=False,
        notes="All three share one wrapper and one parameter namespace. "
              "Implemented natively on scikit-learn rather than via arboreto: "
              "arboreto needs dask+distributed, whose pinned LocalCluster call "
              "was removed upstream, and pinning back pulls numpy<2 - which "
              "conflicts with the numpy 2.x torch requires. Verified on DREAM4 "
              "net3: RF AUROC 0.742 / AUPR 0.125 at ntrees=1000.",
    ))

register(AlgorithmSpec(
    name="KBOOST", runtime=_R, script="r/KBOOST.R",
    library="KBoost", install="BiocManager::install('KBoost')",
    orientation=_SR, family="tree_ensemble",
))

# --- Regression ------------------------------------------------------------
register(AlgorithmSpec(
    name="TIGRESS", runtime=_R, script="r/TIGRESS.R",
    library="tigress", install="devtools::install_github('jpvert/tigress')",
    orientation=_SR, family="regression",
    notes="Not on CRAN. Needs devtools + a working compiler toolchain. "
          "Runs 5 replicates internally and takes the median. Slowest R method.",
))

register(AlgorithmSpec(
    name="INFERELATOR", runtime=_PY, script="python/INFERELATOR.py",
    library="inferelator", install="pip install inferelator",
    orientation=_GR, family="regression",
))

# --- Other -----------------------------------------------------------------
register(AlgorithmSpec(
    name="LEAP", runtime=_R, script="r/LEAP.R",
    library="LEAP", install="install.packages('LEAP')",
    orientation=_GR, family="other",
    notes="Designed for pseudotime-ordered data. On unordered steady-state data "
          "it is very permissive - historically the densest output of the 16.",
))

register(AlgorithmSpec(
    name="MEOMI", runtime=_R, script="r/MEOMI.R",
    library="MEOMI (vendored source)", install="see THIRD_PARTY_NOTICES.md",
    orientation=_GR, family="other",
    notes="Ships its own src/ tree rather than a packaged library. Heaviest "
          "dependency set of any R method (sqldf, igraph, Rgraphviz, ...).",
))

register(AlgorithmSpec(
    name="NONLINEARODES", runtime=_ISO, script="python/NONLINEARODES.py",
    library="xgboost==0.82 + scikit-learn==0.24.2 + numpy==1.16.3",
    install="isolated env only - see notes",
    orientation=_GR, family="other",
    docker_image="adriansegura99/geneci_infer-network_nonlinearodes:4.0.0",
    notes="HARD CONSTRAINT: pinned to Python 3.6 with numpy 1.16.3. This cannot "
          "share an interpreter with PyTorch/PyG. Always runs isolated.",
))


# --------------------------------------------------------------------------
# Additional algorithms (not part of GENECI). MultiGCN ships these wrappers.
# --------------------------------------------------------------------------

register(AlgorithmSpec(
    name="DYNGENIE3", runtime=_PY, script="python/DYNGENIE3.py",
    library="dynGENIE3 (vendored)", install="clone github.com/vahuynh/dynGENIE3",
    orientation=_SR, family="tree_ensemble",
    requires=DataRequirement.TIME_SERIES,
    param_prefix="dyngenie3", vendored=False,
    params={
        "tree_method": ParamSpec("tree_method", "RF", str, choices=["RF", "ET"],
                                 help="Random Forest or Extra-Trees."),
        "K": ParamSpec("K", "sqrt", str,
                       help="Attributes per split: 'sqrt', 'all', or an integer."),
        "ntrees": ParamSpec("ntrees", 1000, int, minimum=1),
        "alpha": ParamSpec("alpha", "from_data", str,
                           help="Degradation rate: 'from_data' or a positive number."),
        "nthreads": ParamSpec("nthreads", 1, int, minimum=1),
    },
    notes="TIME SERIES REQUIRED. Models dY/dt, so steady-state data alone will "
          "not run. Optionally accepts steady-state data *in addition* via "
          "--ss-data. Distinct from GENIE3 despite the name.",
))

register(AlgorithmSpec(
    name="DYNUGENE", runtime=_R, script="r/DYNUGENE.R",
    library="dynUGENE (vendored)", install="devtools::install_local('path/to/dynUGENE')",
    orientation=_SR, family="tree_ensemble",
    requires=DataRequirement.EITHER,
    param_prefix="dynugene", vendored=False,
    params={
        "ntree": ParamSpec("ntree", 10, int, minimum=1,
                           help="Trees per forest. Package default is only 10 - "
                                "raise it for benchmark runs."),
        "mtry": ParamSpec("mtry", 0, int, minimum=0,
                          help="Candidates per split. 0 = package default (p/3)."),
        "alpha": ParamSpec("alpha", "fromData", str),
        "seed": ParamSpec("seed", 777, int),
        "multipleExp": ParamSpec("multipleExp", False, bool,
                                 help="Time-series mode only: data holds several "
                                      "concatenated experiments."),
    },
    notes="Has both inferNetwork (time series) and inferSSNetwork (steady state); "
          "the wrapper picks based on whether --time-series is supplied.",
))

register(AlgorithmSpec(
    name="NETWORKBMA", runtime=_R, script="r/NETWORKBMA.R",
    library="networkBMA", install="BiocManager::install('networkBMA')",
    orientation=_SR, family="bayesian",
    requires=DataRequirement.TIME_SERIES,
    param_prefix="networkbma", vendored=False,
    params={
        "ordering": ParamSpec("ordering", "bic1", str,
                              choices=["bic1", "bic1+prior", "prior"],
                              help="Variable ordering. 'bic1+prior' needs prior.prob."),
        "nvar": ParamSpec("nvar", 50, int, minimum=1,
                          help="Variables retained per gene before BMA."),
        "self": ParamSpec("self", True, bool, help="Allow self edges."),
        "thresProbne0": ParamSpec("thresProbne0", 5, int, minimum=1,
                                  help="iBMAcontrolLM posterior-inclusion threshold (%)."),
    },
    notes="TIME SERIES REQUIRED. Returns posterior probabilities in [0,1], which "
          "is already the exact 3-column continuous format the pipeline wants - "
          "no rescaling needed.",
))

register(AlgorithmSpec(
    name="BNLEARN", runtime=_R, script="r/BNLEARN.R",
    library="bnlearn", install="install.packages('bnlearn')",
    orientation=_SR, family="bayesian",
    requires=DataRequirement.EITHER,
    param_prefix="bnlearn", vendored=False,
    params={
        "algorithm": ParamSpec("algorithm", "hc", str,
                               choices=["hc", "tabu", "pc.stable", "gs",
                                        "iamb", "mmhc", "rsmax2"],
                               help="Structure-learning algorithm."),
        "score": ParamSpec("score", "bic-g", str,
                           choices=["bic-g", "bge", "aic-g"],
                           help="Score for score-based algorithms (-g = Gaussian)."),
        "R": ParamSpec("R", 200, int, minimum=2,
                       help="Bootstrap replicates for boot.strength(). This is "
                            "what makes the output continuous - see notes."),
        "restart": ParamSpec("restart", 0, int, minimum=0,
                             help="Random restarts for hc/tabu."),
        "cluster": ParamSpec("cluster", 1, int, minimum=1,
                             help="Parallel workers via the parallel package."),
    },
    notes="Runs boot.strength(), NOT a single structure fit. A single fit returns "
          "a DAG - binary edges - which is useless to a pipeline that ranks by "
          "continuous confidence. Bootstrap model averaging yields edge strengths "
          "in [0,1]. Score is combined as strength * direction so the output is "
          "directed.",
))

# --------------------------------------------------------------------------
# Partial-correlation and dynamical-Bayesian methods.
# Sourced from the time-series GRN survey tables; all are CRAN/Bioconductor R
# packages producing continuous scores, so they need no new infrastructure.
# --------------------------------------------------------------------------

register(AlgorithmSpec(
    name="PPCOR", runtime=_R, script="r/PPCOR.R",
    library="ppcor", install="install.packages('ppcor')",
    orientation=_SR, family="partial_correlation",
    requires=DataRequirement.STEADY_STATE,
    param_prefix="ppcor", vendored=False,
    params={
        "method": ParamSpec("method", "pearson", str,
                            choices=["pearson", "spearman", "kendall"]),
        "statistic": ParamSpec("statistic", "estimate", str,
                               choices=["estimate", "p.value"],
                               help="Score by |partial correlation| or by 1-p."),
    },
    notes="Partial correlation controls for all other genes, so it suppresses "
          "the indirect edges that plain correlation reports. Requires more "
          "samples than genes for a stable inverse; on 100-gene DREAM4 with "
          "~100 conditions this is borderline and the matrix may be singular.",
))

register(AlgorithmSpec(
    name="GENENET", runtime=_R, script="r/GENENET.R",
    library="GeneNet", install="install.packages('GeneNet')",
    orientation=_SR, family="partial_correlation",
    requires=DataRequirement.EITHER,
    param_prefix="genenet", vendored=False,
    params={
        "method": ParamSpec("method", "static", str, choices=["static", "dynamic"],
                            help="'dynamic' uses the time-series ggm.estimate."),
        "statistic": ParamSpec("statistic", "pcor", str,
                               choices=["pcor", "prob"],
                               help="Partial correlation, or posterior edge probability."),
    },
    notes="Shrinkage estimator for graphical Gaussian models, so unlike ppcor it "
          "stays well-defined when genes outnumber samples. That makes it the "
          "safer partial-correlation choice on wide data.",
))

register(AlgorithmSpec(
    name="G1DBN", runtime=_R, script="r/G1DBN.R",
    library="G1DBN", install="install.packages('G1DBN')",
    orientation=_SR, family="dynamic_bayesian",
    requires=DataRequirement.TIME_SERIES,
    param_prefix="g1dbn", vendored=False,
    params={
        "alpha1": ParamSpec("alpha1", 0.5, float, minimum=0.0, maximum=1.0,
                            help="Step-1 edge-selection threshold (deliberately loose)."),
        "predPosition": ParamSpec("predPosition", 0, int, minimum=0,
                                  help="0 = all genes are candidate predictors."),
    },
    notes="TIME SERIES REQUIRED. Two-step dynamic Bayesian network: step 1 scores "
          "every pair with a first-order DBN, step 2 refits on the survivors. "
          "Score is 1 - the step-2 edge p-value.",
))

register(AlgorithmSpec(
    name="SCANBMA", runtime=_R, script="r/SCANBMA.R",
    library="networkBMA", install="BiocManager::install('networkBMA')",
    orientation=_SR, family="dynamic_bayesian",
    requires=DataRequirement.TIME_SERIES,
    param_prefix="scanbma", vendored=False,
    params={
        "g": ParamSpec("g", 20.0, float, minimum=0.0,
                       help="Zellner g-prior parameter."),
        "odds_ratio": ParamSpec("odds_ratio", 10000.0, float, minimum=1.0,
                                help="Occam window width for model pruning."),
        "nvar": ParamSpec("nvar", 50, int, minimum=1),
    },
    notes="Free to add: ships in networkBMA, already a dependency. Searches model "
          "space directly rather than the iterative BMA path NETWORKBMA uses, so "
          "the two give genuinely different edge sets from one install.",
))

register(AlgorithmSpec(
    name="TDARACNE", runtime=_R, script="r/TDARACNE.R",
    library="TDARACNE", install="BiocManager::install('TDARACNE')",
    orientation=_GR, family="mutual_information",
    requires=DataRequirement.TIME_SERIES,
    param_prefix="tdaracne", vendored=False,
    params={
        "delta": ParamSpec("delta", 3, int, minimum=1,
                           help="Maximum time delay considered."),
        "likelihood": ParamSpec("likelihood", 1.2, float, minimum=1.0,
                                help="Likelihood ratio threshold."),
        "norm": ParamSpec("norm", 1, int, choices=[0, 1, 2]),
        "thresh": ParamSpec("thresh", 0.0, float, minimum=0.0),
    },
    notes="TIME SERIES REQUIRED. Time-delayed ARACNE: infers direction from which "
          "gene changes first, which plain ARACNE cannot do. Needs an "
          "ExpressionSet, which the wrapper builds. Verify availability - it has "
          "been deprecated in some Bioconductor releases.",
))

register(AlgorithmSpec(
    name="SWING", runtime=_PY, script="python/SWING.py",
    library="Swing (local clone)", install="clone github.com/bagherilab/SWING",
    orientation=_SR, family="granger_causality",
    requires=DataRequirement.TIME_SERIES,
    param_prefix="swing", vendored=False,
    params={
        "window_type": ParamSpec("window_type", "RandomForest", str,
                                 choices=["RandomForest", "Lasso", "Dionesus"],
                                 help="Per-window regression model."),
        "min_lag": ParamSpec("min_lag", 1, int, minimum=0,
                             help="Smallest time lag considered."),
        "max_lag": ParamSpec("max_lag", 3, int, minimum=1,
                             help="Largest time lag considered."),
        "window_width": ParamSpec("window_width", 10, int, minimum=2,
                                  help="Time points per sliding window."),
        "n_trees": ParamSpec("n_trees", 500, int, minimum=1,
                             help="Trees per window (RandomForest only). The "
                                  "SWING example uses 10, which is very low."),
        "n_jobs": ParamSpec("n_jobs", -1, int,
                            help="-1 uses all cores."),
        "sort_by": ParamSpec("sort_by", "mean", str,
                             choices=["mean", "median", "rank"],
                             help="Statistic used to aggregate across windows."),
        "source_path": ParamSpec("source_path", "", str,
                                 help="Path to the SWING clone, if not on "
                                      "sys.path or in SWING_PATH."),
    },
    notes="TIME SERIES REQUIRED. Granger causality via sliding windows - the "
          "only method here that tests whether one gene's past improves "
          "prediction of another's future, so it adds a family nothing else "
          "covers. Its requirements.txt pins 2016-era versions, but setup.py "
          "leaves them unpinned and the only API actually removed since is "
          "DataFrame.append; the wrapper shims that, so it runs in-process.",
))

# --------------------------------------------------------------------------
# GENECI techniques that ship as prebuilt Docker images only.
#
# MATLAB tools are compiled against the MATLAB Compiler Runtime (free, no
# licence) and published to Docker Hub, so no MATLAB install is needed - only
# `docker pull`. Julia tools are containerised the same way. This is the same
# mechanism NONLINEARODES uses; there is nothing special about these beyond
# needing Docker.
#
# On Apple Silicon these are linux/amd64 images and run under emulation. Set
# Config.docker_platform="linux/amd64" (the default) and expect them to be
# several times slower than native.
# --------------------------------------------------------------------------

_GENECI_TAG = "4.0.0"


def _geneci_image(name: str) -> str:
    return f"adriansegura99/geneci_infer-network_{name.lower()}:{_GENECI_TAG}"


for _name, _family, _orient, _req, _note in [
    ("JUMP3", "regression", _GR, DataRequirement.TIME_SERIES,
     "MATLAB/MCR. On-off dynamical model of promoter states; the only method "
     "here that models transcription as a switching process."),
    ("NARROMI", "regression", _GR, DataRequirement.STEADY_STATE,
     "MATLAB/MCR. Noise-and-redundancy reduction: linear programming to remove "
     "noise, then mutual information to prune redundant edges."),
    ("GRNVBEM", "bayesian", _GR, DataRequirement.TIME_SERIES,
     "MATLAB/MCR. Variational Bayesian EM on a first-order autoregressive "
     "model. Distinct from the BMA methods - approximates the posterior "
     "rather than averaging over sampled models."),
    ("PLSNET", "regression", _GR, DataRequirement.STEADY_STATE,
     "MATLAB/MCR. Partial least squares feature selection with ensemble "
     "aggregation. Handles collinear predictors better than plain regression."),
    ("CMI2NI", "mutual_information", _GR, DataRequirement.STEADY_STATE,
     "MATLAB/MCR. Conditional mutual inclusive information - corrects the "
     "underestimation bias of standard conditional MI."),
    ("PCACMI", "mutual_information", _GR, DataRequirement.STEADY_STATE,
     "MATLAB/MCR. Path consistency algorithm driven by conditional MI."),
    ("LOCPCACMI", "mutual_information", _GR, DataRequirement.STEADY_STATE,
     "MATLAB/MCR. Local variant of PCACMI: applies the path consistency "
     "search within local neighbourhoods, which scales better."),
    ("RSNET", "regression", _GR, DataRequirement.STEADY_STATE,
     "MATLAB/MCR. Recursive selection with elastic-net style regularisation."),
    ("PIDC", "information_decomposition", _GR, DataRequirement.STEADY_STATE,
     "Julia. Partial information decomposition - separates unique, redundant "
     "and synergistic contributions of gene pairs, which pairwise MI cannot "
     "distinguish. A genuinely different family, and a strong performer in "
     "the BEELINE benchmarks."),
    ("PUC", "information_decomposition", _GR, DataRequirement.STEADY_STATE,
     "Julia. Proportional unique contribution - the pairwise component "
     "underlying PIDC, exposed separately."),
]:
    register(AlgorithmSpec(
        name=_name,
        runtime=Runtime.DOCKER,
        script=None,
        library="GENECI container",
        install=f"docker pull {_geneci_image(_name)}",
        orientation=_orient,
        family=_family,
        requires=_req,
        docker_image=_geneci_image(_name),
        notes=_note + " No tunable parameters: the container entrypoint takes "
                      "only (input, output).",
    ))

# --------------------------------------------------------------------------
# Convenience accessors
# --------------------------------------------------------------------------

DEFAULT_METHODS: List[str] = list(ALGORITHMS.keys())


def resolve(methods=None, all_methods: bool = False) -> List[AlgorithmSpec]:
    """
    Turn a user-supplied method selection into a list of specs.

    Fails loudly on an unknown name rather than silently skipping it - a silently
    dropped algorithm looks identical to a bad result in the fused output.
    """
    if all_methods:
        return [ALGORITHMS[n] for n in DEFAULT_METHODS]
    if not methods:
        raise ValueError(
            "No methods selected. Pass methods=[...] or all_methods=True. "
            f"Available: {', '.join(DEFAULT_METHODS)}"
        )
    if isinstance(methods, str):
        methods = [methods]

    unknown = [m for m in methods if m not in ALGORITHMS]
    if unknown:
        raise ValueError(
            f"Unknown method(s): {', '.join(unknown)}. "
            f"Available: {', '.join(DEFAULT_METHODS)}"
        )
    return [ALGORITHMS[m] for m in methods]


def by_family() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for spec in ALGORITHMS.values():
        out.setdefault(spec.family, []).append(spec.name)
    return out


def requiring(req: "DataRequirement") -> List[str]:
    return [s.name for s in ALGORITHMS.values() if s.requires == req]


def by_runtime() -> Dict[Runtime, List[str]]:
    out: Dict[Runtime, List[str]] = {}
    for spec in ALGORITHMS.values():
        out.setdefault(spec.runtime, []).append(spec.name)
    return out
