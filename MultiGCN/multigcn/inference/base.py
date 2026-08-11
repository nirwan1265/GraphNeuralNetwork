"""Runner abstraction: one interface, several execution backends."""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Optional

from ..registry import AlgorithmSpec


class RunnerError(RuntimeError):
    """Raised when an algorithm fails to execute or produces no output."""


class BaseRunner(abc.ABC):
    """
    Executes one algorithm against one expression file, producing
    <output_dir>/GRN_<NAME>.csv - a headerless 3-column edge list
    (source, target, continuous_confidence).
    """

    def __init__(self, spec: AlgorithmSpec, params: Optional[dict] = None,
                 time_series: Optional[Path] = None):
        self.spec = spec
        self.params = params or {}
        # Resolve to absolute. RScriptRunner executes with cwd set to the
        # scripts directory so wrappers can source their siblings, which means
        # a relative path handed in here resolves against the wrong directory.
        # Python runners inherit the caller's cwd, so this failed for R
        # algorithms only - a confusing asymmetry.
        self.time_series = Path(time_series).resolve() if time_series else None

    @property
    def params_json(self) -> str:
        import json
        return json.dumps(self.params) if self.params else ""

    @abc.abstractmethod
    def available(self) -> bool:
        """True if this runner can actually execute on this machine, right now."""

    @abc.abstractmethod
    def _execute(self, expression_file: Path, output_dir: Path) -> None:
        ...

    def _check_data_requirement(self) -> None:
        from ..registry import DataRequirement
        if self.spec.requires is DataRequirement.TIME_SERIES and not self.time_series:
            raise RunnerError(
                f"{self.spec.name} requires time-series data and none was given.\n"
                f"It models expression change between consecutive time points, so "
                f"steady-state data cannot substitute.\n"
                f"Pass time_series='path/to/timeseries.csv' (first column 'time')."
            )

    def run(self, expression_file: Path, output_dir: Path,
            overwrite: bool = False) -> Path:
        self._check_data_requirement()
        expression_file = Path(expression_file).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / self.spec.output_filename

        if target.exists() and not overwrite:
            return target  # cached; inference is expensive, never redo it silently

        self._execute(expression_file, output_dir)

        # Fail loudly. A missing or empty output must never be mistaken for
        # "this algorithm found nothing" - that silently corrupts the fusion.
        if not target.exists():
            raise RunnerError(
                f"{self.spec.name} finished but produced no {target.name}. "
                f"Expected it in {output_dir}."
            )
        if target.stat().st_size == 0:
            raise RunnerError(f"{self.spec.name} produced an empty {target.name}.")
        return target
