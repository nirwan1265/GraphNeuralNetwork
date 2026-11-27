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
    dependencies cannot coexist with MultiGCN's.

    Images are pulled automatically on first use. Nothing to install by hand -
    Docker itself is the only prerequisite.
    """
    platform: str = "linux/amd64"

    def available(self) -> bool:
        return shutil.which("docker") is not None

    def workdir(self) -> str:
        """
        The image's WORKDIR, or '/' if it declares none.

        GENECI's containers are built to be invoked from their WORKDIR with
        *relative* arguments - the entrypoints are relative script paths, and
        several wrappers write to `./{output_folder}/GRN_X.csv`. Handing them
        an absolute '/data_out' makes that './/data_out', which resolves
        against the WORKDIR and does not exist. So mount inside the WORKDIR
        and pass a relative path, exactly as GENECI does.
        """
        proc = subprocess.run(
            ["docker", "image", "inspect", self.spec.docker_image,
             "--format", "{{.Config.WorkingDir}}"],
            capture_output=True, text=True)
        wd = proc.stdout.strip() if proc.returncode == 0 else ""
        return wd.rstrip("/") or ""

    def image_present(self) -> bool:
        proc = subprocess.run(
            ["docker", "image", "inspect", self.spec.docker_image],
            capture_output=True, text=True)
        return proc.returncode == 0

    def pull(self, quiet: bool = False) -> None:
        """
        Pull the image, streaming progress.

        `docker run` would pull implicitly, but it does so silently - a first
        run then looks like a multi-GB hang with no output. Pulling explicitly
        makes the wait legible and separates a download failure from an
        algorithm failure.
        """
        if self.image_present():
            return
        cmd = ["docker", "pull"]
        if self.platform:
            cmd += ["--platform", self.platform]
        cmd.append(self.spec.docker_image)

        if not quiet:
            print(f"    pulling {self.spec.docker_image} (first use only) ...")
        proc = subprocess.run(cmd, capture_output=quiet, text=True)
        if proc.returncode != 0:
            raise RunnerError(
                f"Could not pull {self.spec.docker_image}.\n"
                f"Check that Docker is running and you have network access.\n"
                + (proc.stderr[-2000:] if proc.stderr else "")
            )

    def _execute(self, expression_file: Path, output_dir: Path) -> None:
        if not self.available():
            raise RunnerError(
                f"{self.spec.name} runs as a container, but `docker` is not on "
                f"PATH. Install Docker Desktop and start it."
            )
        if not self.spec.docker_image:
            raise RunnerError(f"No docker_image registered for {self.spec.name}.")

        self.pull()

        work = output_dir / "_docker_in"
        work.mkdir(parents=True, exist_ok=True)
        staged = work / expression_file.name
        shutil.copyfile(expression_file, staged)

        # Mount relative to the image's WORKDIR and pass relative paths, so
        # that wrappers doing './{output_folder}' resolve correctly. See
        # workdir() for why absolute paths fail.
        wd = self.workdir()
        in_mount = f"{wd}/data_in"
        out_mount = f"{wd}/data_out"

        cmd = ["docker", "run", "--rm"]
        if self.platform:
            cmd += ["--platform", self.platform]
        cmd += [
            "-v", f"{work}:{in_mount}", "-v", f"{output_dir}:{out_mount}",
            self.spec.docker_image,
            f"data_in/{expression_file.name}", "data_out",
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
