"""Concrete runners: local R, local Python, isolated Python, Docker."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .base import BaseRunner, RunnerError
from ..registry import AlgorithmSpec, Runtime

SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _script_path(spec: AlgorithmSpec) -> Path:
    if spec.script is None:
        raise RunnerError(
            f"{spec.name} has no wrapper script - it runs as a container only.")
    p = SCRIPTS_DIR / spec.script
    if not p.exists():
        raise RunnerError(
            f"Wrapper script missing: {p}\n"
            f"Run `python -m multigcn.inference.vendor --geneci-path /path/to/GENECI` "
            f"to populate the vendored scripts."
        )
    return p


def _run(cmd, cwd=None, name="") -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RunnerError(
            f"{name} exited with code {proc.returncode}.\n"
            f"--- command ---\n{' '.join(map(str, cmd))}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )


class RScriptRunner(BaseRunner):
    def available(self) -> bool:
        return shutil.which("Rscript") is not None

    def _execute(self, expression_file: Path, output_dir: Path) -> None:
        if not self.available():
            raise RunnerError(
                f"{self.spec.name} needs R, but `Rscript` is not on PATH.\n"
                f"Install R, then in R: {self.spec.install}"
            )
        script = _script_path(self.spec)
        cmd = ["Rscript", "--vanilla", str(script),
               str(expression_file), str(output_dir)]
        if self.params or self.time_series:
            cmd.append(self.params_json)
        if self.time_series:
            cmd.append(str(self.time_series))
        _run(cmd, cwd=SCRIPTS_DIR.parent, name=self.spec.name)


class PythonRunner(BaseRunner):
    def available(self) -> bool:
        return True

    def _execute(self, expression_file: Path, output_dir: Path) -> None:
        script = _script_path(self.spec)
        cmd = [sys.executable, str(script), str(expression_file), str(output_dir)]
        cmd += list(self.spec.extra_args)
        if self.time_series:
            cmd.append(str(self.time_series))
        if self.params:
            cmd.append(self.params_json)
        _run(cmd, name=self.spec.name)


class DockerRunner(BaseRunner):
    """
    For algorithms that only exist as containers: the MATLAB tools compiled
    against the MATLAB Compiler Runtime, the Julia tools, and anything whose
    dependencies cannot coexist with MultiGCN's (NONLINEARODES).
    """
    platform: str = "linux/amd64"

    def available(self) -> bool:
        return shutil.which("docker") is not None

    def _execute(self, expression_file: Path, output_dir: Path) -> None:
        if not self.available():
            raise RunnerError(
                f"{self.spec.name} requires Docker (incompatible dependencies: "
                f"{self.spec.library}), but `docker` is not on PATH."
            )
        if not self.spec.docker_image:
            raise RunnerError(f"No docker_image registered for {self.spec.name}.")

        work = output_dir / "_docker_in"
        work.mkdir(parents=True, exist_ok=True)
        staged = work / expression_file.name
        shutil.copyfile(expression_file, staged)

        cmd = ["docker", "run", "--rm"]
        if self.platform:
            cmd += ["--platform", self.platform]
        cmd += [
            "-v", f"{work}:/data_in", "-v", f"{output_dir}:/data_out",
            self.spec.docker_image,
            f"/data_in/{expression_file.name}", "/data_out",
        ] + list(self.spec.extra_args)
        _run(cmd, name=self.spec.name)
        shutil.rmtree(work, ignore_errors=True)


RUNNERS = {
    Runtime.R: RScriptRunner,
    Runtime.PYTHON: PythonRunner,
    Runtime.PYTHON_ISOLATED: DockerRunner,
    Runtime.DOCKER: DockerRunner,
}


def get_runner(spec: AlgorithmSpec, prefer_docker: bool = False,
               params=None, time_series=None,
               docker_platform: str = "linux/amd64") -> BaseRunner:
    if prefer_docker and spec.docker_image:
        runner = DockerRunner(spec, params, time_series)
        runner.platform = docker_platform
        return runner
    cls = RUNNERS.get(spec.runtime)
    if cls is None:
        raise RunnerError(f"No runner for runtime {spec.runtime}")
    runner = cls(spec, params, time_series)
    if isinstance(runner, DockerRunner):
        runner.platform = docker_platform
    return runner
