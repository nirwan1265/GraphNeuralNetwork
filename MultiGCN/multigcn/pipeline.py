"""Top-level MultiGCN pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .config import Config
from .data.loaders import (DataError, load_algorithm_matrices, load_edge_list,
                           load_prior_network, read_expression)
from .evaluation.metrics import binary_metrics, ranking_metrics
from .fusion.combine import agreement_weights, blend_with_inputs, fuse
from .inference.runners import get_runner
from .postprocessing.tukey import tukey_cutoff
from .params import ParamError, describe, resolved_for, route
from .registry import ALGORITHMS, AlgorithmSpec, DataRequirement, resolve
from .topology.threshold import (density, edge_index_from_adj,
                                normalized_laplacian_edge_index,
                                top_x_percent_edges)


class MultiGCN:
    """
    Combine many GRN inference algorithms through a GCN graph autoencoder.

        model = MultiGCN(methods=["GENIE3_ET", "CLR", "ARACNE"], top_x=0.10)
        model.infer("expression.csv")          # run the algorithms
        scores = model.fit_predict()           # train + fuse
        model.evaluate("gold_standard.tsv")

    Or, for everything:

        model = MultiGCN(all_methods=True)
    """

    def __init__(
        self,
        methods: Optional[Union[str, List[str]]] = None,
        all_methods: bool = False,
        config: Optional[Config] = None,
        **kwargs,
    ):
        self.config = config or Config()
        self.specs: List[AlgorithmSpec] = resolve(methods, all_methods)

        # Split <package>_<param> kwargs from Config kwargs, validating as we go.
        self.algorithm_params, config_kwargs = route(kwargs, self.specs)
        for k, v in config_kwargs.items():
            if not hasattr(self.config, k):
                raise TypeError(
                    f"Unknown option {k!r}. Algorithm parameters must be "
                    f"prefixed with their package, e.g. genie3_ntrees=1000."
                )
            setattr(self.config, k, v)

        self.time_series: Optional[Path] = None
        self.priors: Dict[str, np.ndarray] = {}

        self.genes: Optional[List[str]] = None
        self.matrices: Dict[str, np.ndarray] = {}
        self.reconstructions: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.fused: Optional[np.ndarray] = None
        self.model_output: Optional[np.ndarray] = None   # before the input blend
        self.weights: Optional[Dict[str, float]] = None

        self.output_dir = Path(self.config.output_dir)
        self.lists_dir = self.output_dir / "lists"

    # -- reporting ---------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg)

    @property
    def method_names(self) -> List[str]:
        return [s.name for s in self.specs] + list(self.priors)

    def params_for(self, name: str) -> Dict:
        """Resolved parameters (defaults + user overrides) for one algorithm."""
        return resolved_for(ALGORITHMS[name], self.algorithm_params)

    def describe_params(self) -> str:
        """Every tunable parameter of every selected algorithm."""
        blocks = [describe(s) for s in self.specs if s.tunable]
        untunable = [s.name for s in self.specs if not s.tunable]
        if untunable:
            blocks.append(
                "No tunable parameters (GENECI wrappers hardcode their settings): "
                + ", ".join(untunable))
        return "\n\n".join(blocks)

    def plan(self) -> str:
        """Human-readable summary of what will run, and with which runtime."""
        lines = [f"MultiGCN: {len(self.specs)} algorithm(s)"]
        for s in self.specs:
            runner = get_runner(s, self.config.prefer_docker)
            ok = "ok" if runner.available() else "UNAVAILABLE"
            need = "" if s.requires is DataRequirement.STEADY_STATE \
                else f"  <needs {s.requires.value}>"
            over = self.algorithm_params.get(s.param_prefix or "", {})
            tuned = f"  ({', '.join(f'{k}={v}' for k, v in over.items())})" if over else ""
            lines.append(
                f"  {s.name:<14} {s.runtime.value:<16} {s.library:<12} [{ok}]{need}{tuned}")
        for name in self.priors:
            lines.append(f"  {name:<14} {'prior':<16} {'(curated)':<12} [ok]")
        return "\n".join(lines)

    # -- step 1: inference -------------------------------------------------
    def infer(self, expression_file, time_series=None,
              skip_failures: bool = False) -> Path:
        """
        Run each selected algorithm, writing GRN_<NAME>.csv into lists/.

        time_series: optional CSV whose first column is 'time'. Required by
        DYNGENIE3 and NETWORKBMA; optional for DYNUGENE (switches it from
        steady-state to time-series mode).
        """
        expression_file = Path(expression_file)
        self.time_series = Path(time_series) if time_series else None

        needs_ts = [s.name for s in self.specs
                    if s.requires is DataRequirement.TIME_SERIES]
        if needs_ts and not self.time_series:
            raise ValueError(
                f"{', '.join(needs_ts)} require time-series data. "
                f"Pass time_series='...csv' (first column 'time'), or drop them "
                f"from the method list."
            )
        expr = read_expression(expression_file)
        self.genes = list(expr.index)
        self._log(f"Expression: {len(self.genes)} genes x {expr.shape[1]} samples")
        self.lists_dir.mkdir(parents=True, exist_ok=True)

        failures = {}
        for spec in self.specs:
            runner = get_runner(
                spec, self.config.prefer_docker,
                params=resolved_for(spec, self.algorithm_params),
                time_series=self.time_series,
                docker_platform=self.config.docker_platform,
            )
            # Say plainly whether this is real work or a cache hit. Printing
            # "running" for an instant cache reuse makes a 20-minute run and a
            # 2-second one look identical in the log.
            cached = (self.lists_dir / spec.output_filename).exists()
            if cached and not self.config.overwrite_inference:
                self._log(f"  {spec.name}: cached, reusing")
            else:
                self._log(f"  running {spec.name} ({spec.runtime.value}) ...")
            try:
                runner.run(expression_file, self.lists_dir,
                           overwrite=self.config.overwrite_inference)
            except Exception as e:
                if not skip_failures:
                    raise
                failures[spec.name] = str(e)
                self._log(f"    FAILED: {e}")

        if failures:
            self.specs = [s for s in self.specs if s.name not in failures]
            self._log(f"  continuing with {len(self.specs)} algorithm(s); "
                      f"failed: {', '.join(failures)}")
        return self.lists_dir

    def add_prior(self, name: str, path, symmetric: bool = False) -> None:
        """
        Add a known/curated network as an extra input channel.

        Keep the source distinct from your evaluation ground truth, or you are
        feeding the answer into the input.
        """
        if self.genes is None:
            raise RuntimeError("Call infer() or load() first so gene order is known.")
        if name in ALGORITHMS:
            raise ValueError(f"{name!r} collides with a registered algorithm name.")
        self.priors[name] = load_prior_network(path, self.genes, symmetric)

    # -- step 2: load ------------------------------------------------------
    def load(self, lists_dir=None, expression_file=None) -> Dict[str, np.ndarray]:
        """Load precomputed edge lists, skipping inference entirely."""
        if expression_file is not None:
            self.genes = list(read_expression(expression_file).index)
        if self.genes is None:
            raise RuntimeError("Gene order unknown - pass expression_file=.")

        requested = [s.name for s in self.specs]
        lists_dir = Path(lists_dir or self.lists_dir)

        # Load one file at a time. A single malformed edge list must not
        # discard an entire inference run that may have taken hours - but it
        # must also never be swallowed silently, so each rejection is reported
        # by name and reason.
        self.matrices = {}
        rejected = {}
        for name in requested:
            path = lists_dir / f"GRN_{name}.csv"
            if not path.exists():
                rejected[name] = "no edge list produced"
                continue
            try:
                self.matrices[name] = load_edge_list(path, self.genes)
            except DataError as exc:
                rejected[name] = str(exc).split(". ")[0]

        if rejected:
            self._log("  excluded from fusion:")
            for name, why in rejected.items():
                self._log(f"    {name}: {why}")

        if not self.matrices:
            raise DataError(
                f"No usable edge lists in {lists_dir}. "
                f"Rejected: {', '.join(rejected) or 'none found'}"
            )

        self.matrices.update(self.priors)
        return self.matrices

    # -- steps 3-6: train and fuse ----------------------------------------
    def fit_predict(self) -> np.ndarray:
        """
        Train and fuse.

        model="joint" (default): one multi-relational GAE over every
        algorithm's top-x graph. model="separate": one GAE per algorithm.
        Reconstructions are combined with agreement weights (or flat), then
        blended with the rank-average of the raw scores (input_blend).
        """
        from .training.train import train_gae, train_joint  # lazy: torch only

        if not self.matrices:
            self.load()
        cfg = self.config
        if cfg.model not in ("joint", "separate"):
            raise ValueError(f"model must be 'joint' or 'separate', got {cfg.model!r}")
        if cfg.fusion_weights not in ("agreement", "flat"):
            raise ValueError("fusion_weights must be 'agreement' or 'flat'")
        n = len(self.genes)

        adjs = {}
        for name, mat in self.matrices.items():
            adjs[name] = top_x_percent_edges(mat, cfg.top_x, cfg.keep_weights,
                                             symmetric=cfg.symmetric)
            self._log(f"  {name:<14} input density {density(adjs[name])*100:5.2f}%")

        self.reconstructions = {}
        if cfg.model == "joint":
            if not cfg.symmetric:
                raise ValueError("model='joint' scores undirected pairs; "
                                 "use symmetric=True or model='separate'.")
            self._log(f"  training one joint model over {len(adjs)} algorithms "
                      f"x {cfg.iterations} seed(s)")
            runs = []
            for it in range(cfg.iterations):
                seed = None if cfg.seed is None else cfg.seed + it
                runs.append(train_joint(
                    adjs, epochs=cfg.epochs, lr=cfg.lr,
                    weight_decay=cfg.weight_decay, dim=cfg.embedding_dim,
                    num_bases=cfg.joint_bases, dropout=cfg.dropout,
                    seed=seed, device=cfg.device))
            self.reconstructions = {a: {"joint": [r[a] for r in runs]} for a in adjs}
        else:
            for name, adj in adjs.items():
                edge_index = (normalized_laplacian_edge_index(adj) if cfg.laplacian
                              else edge_index_from_adj(adj))
                self.reconstructions[name] = {}
                for dec in cfg.decoders:
                    runs = []
                    for it in range(cfg.iterations):
                        seed = None if cfg.seed is None else cfg.seed + it
                        runs.append(train_gae(
                            edge_index, n, decoder=dec, epochs=cfg.epochs,
                            lr=cfg.lr, weight_decay=cfg.weight_decay,
                            hidden=cfg.hidden, out_channels=cfg.embedding_dim,
                            dropout=cfg.dropout, seed=seed, device=cfg.device))
                    self.reconstructions[name][dec] = runs

        self.weights = (agreement_weights(self.matrices, cfg.agreement_power,
                                          symmetric=cfg.symmetric)
                        if cfg.fusion_weights == "agreement" else None)
        total = sum(len(r) for d in self.reconstructions.values() for r in d.values())
        self._log(f"  fusing {total} reconstructions ({cfg.fusion_weights} weights"
                  f"{f', {cfg.input_blend:.0%} rank-average blend' if cfg.input_blend else ''})")
        self.model_output = fuse(self.reconstructions, self.weights)
        self.fused = blend_with_inputs(self.model_output, self.matrices,
                                       cfg.input_blend, self.weights,
                                       symmetric=cfg.symmetric)
        return self.fused

    # -- step 7: optional extraction --------------------------------------
    def extract_network(self, k: float = 1.5) -> np.ndarray:
        if self.fused is None:
            raise RuntimeError("Call fit_predict() first.")
        return tukey_cutoff(self.fused, k=k)

    # -- evaluation --------------------------------------------------------
    def evaluate(self, gold_standard, k: Optional[float] = None) -> Dict[str, float]:
        from .evaluation.metrics import load_gold_standard
        if self.fused is None:
            raise RuntimeError("Call fit_predict() first.")
        truth = load_gold_standard(gold_standard, self.genes)
        results = ranking_metrics(self.fused, truth,
                                  undirected=self.config.undirected_eval)
        if k is not None:
            results.update(binary_metrics(
                self.extract_network(k), truth,
                undirected=self.config.undirected_eval))
        return results

    def save(self, path=None) -> Path:
        if self.fused is None:
            raise RuntimeError("Nothing to save - call fit_predict() first.")
        path = Path(path or self.output_dir / "fused_network.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        pd.DataFrame(self.fused, index=self.genes, columns=self.genes).to_csv(path)
        return path

    def save_matrices(self, directory=None, include_reconstructions: bool = True):
        """
        Write one weight matrix per algorithm as a labelled genes x genes CSV.

        Two kinds of matrix are produced, and they are not interchangeable:

          matrices/input/<ALGO>.csv
              the algorithm's own raw confidence scores, before any
              thresholding - what that algorithm alone believes

          matrices/reconstructed/<ALGO>__<decoder>.csv
              the GCN's reconstruction after training on that algorithm's
              top-X%% topology - what the model believes having seen it

        Comparing the two is how you see what the GCN actually added.
        """
        import numpy as np
        import pandas as pd

        if not self.matrices:
            raise RuntimeError("No matrices loaded - call load() or infer() first.")

        root = Path(directory or self.output_dir / "matrices")
        (root / "input").mkdir(parents=True, exist_ok=True)
        written = []

        for name, mat in self.matrices.items():
            p = root / "input" / f"{name}.csv"
            pd.DataFrame(mat, index=self.genes, columns=self.genes).to_csv(p)
            written.append(p)

        if include_reconstructions and self.reconstructions:
            (root / "reconstructed").mkdir(parents=True, exist_ok=True)
            for name, per_decoder in self.reconstructions.items():
                for decoder, runs in per_decoder.items():
                    # Average the iterations so one file = one (algorithm, decoder).
                    avg = np.mean(np.stack(runs, axis=0), axis=0)
                    p = root / "reconstructed" / f"{name}__{decoder}.csv"
                    pd.DataFrame(avg, index=self.genes,
                                 columns=self.genes).to_csv(p)
                    written.append(p)

        if self.fused is not None:
            p = root / "fused.csv"
            pd.DataFrame(self.fused, index=self.genes,
                         columns=self.genes).to_csv(p)
            written.append(p)

        self._log(f"  wrote {len(written)} matrices to {root}")
        return written

    def per_algorithm_scores(self, gold_standard):
        """
        AUROC/AUPR for each algorithm's raw output and for the fused network,
        as a tidy DataFrame. This is the table that shows whether the ensemble
        actually beat its inputs.
        """
        import pandas as pd

        from .evaluation.metrics import load_gold_standard, ranking_metrics

        truth = load_gold_standard(gold_standard, self.genes)
        und = self.config.undirected_eval
        rows = []
        for name, mat in self.matrices.items():
            r = ranking_metrics(mat, truth, undirected=und)
            rows.append({"method": name, "stage": "input", **r})
        for name, per_decoder in self.reconstructions.items():
            import numpy as np
            for decoder, runs in per_decoder.items():
                avg = np.mean(np.stack(runs, axis=0), axis=0)
                r = ranking_metrics(avg, truth, undirected=und)
                rows.append({"method": f"{name}__{decoder}",
                             "stage": "reconstructed", **r})
        if self.model_output is not None and self.config.input_blend:
            rows.append({"method": "MultiGCN_model_only", "stage": "fused",
                         **ranking_metrics(self.model_output, truth, undirected=und)})
        if self.fused is not None:
            rows.append({"method": "MultiGCN_fused", "stage": "fused",
                         **ranking_metrics(self.fused, truth, undirected=und)})
        return pd.DataFrame(rows).sort_values("aupr", ascending=False)
