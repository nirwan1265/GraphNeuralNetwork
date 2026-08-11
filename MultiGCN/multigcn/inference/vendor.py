"""
Copy the wrapper scripts out of a local GENECI checkout into MultiGCN.

We deliberately copy rather than re-type them: the wrappers encode small but
load-bearing details (matrix orientation, per-algorithm rescaling, replicate
counts) that are easy to get subtly wrong by hand, and a subtly wrong wrapper
produces a plausible network that is quietly meaningless.

GENECI is MIT licensed (Copyright (c) 2022 AdrianSeguraOrtiz), so redistributing
these wrappers with attribution is permitted. See THIRD_PARTY_NOTICES.md.

Usage:
    python -m multigcn.inference.vendor --geneci-path ~/Documents/Github/GENECI
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ..registry import ALGORITHMS, Runtime

HERE = Path(__file__).parent
SCRIPTS = HERE / "scripts"

ATTRIBUTION = """\
# ---------------------------------------------------------------------------
# Vendored from GENECI (https://github.com/AdrianSeguraOrtiz/GENECI)
# Copyright (c) 2022 Adrian Segura Ortiz -- MIT License.
# Retrieved and adapted for MultiGCN. See THIRD_PARTY_NOTICES.md.
#
# Local adaptation: the `source(...)` path for shared helpers is rewritten to
# resolve relative to this package rather than GENECI's container WORKDIR.
# ---------------------------------------------------------------------------
"""


def vendor(geneci_path: Path, force: bool = False) -> None:
    src_root = Path(geneci_path).expanduser().resolve() / "components" / "infer_network"
    if not src_root.is_dir():
        raise SystemExit(f"Not a GENECI checkout: {src_root} does not exist.")

    (SCRIPTS / "r").mkdir(parents=True, exist_ok=True)
    (SCRIPTS / "python").mkdir(parents=True, exist_ok=True)

    # Shared R helper (GetConfList / ProcessList) used by every R wrapper.
    helpers = src_root / "functions.R"
    if helpers.exists():
        dst = SCRIPTS / "r" / "functions.R"
        dst.write_text(ATTRIBUTION + helpers.read_text())
        print(f"  functions.R -> {dst.relative_to(HERE.parent.parent)}")

    seen_scripts = set()
    for name, spec in ALGORITHMS.items():
        if spec.script is None or spec.script in seen_scripts:
            continue
        seen_scripts.add(spec.script)

        # NEVER overwrite a MultiGCN-owned wrapper with GENECI's version.
        # GENECI's scripts take exactly (input, output) and hardcode their
        # hyperparameters; ours take a trailing params JSON. Copying theirs
        # over ours breaks the parameter interface in a way that surfaces as a
        # confusing argument error much later, at run time.
        if not spec.vendored:
            print(f"  [skip] {name}: MultiGCN-owned wrapper, not vendored")
            continue

        # GENIE3_RF/ET/GRNBOOST2 all map to GENIE3/GENIE3.py
        stem = Path(spec.script).stem
        folder = "GENIE3" if stem == "GENIE3" else stem
        src = src_root / folder / Path(spec.script).name

        if not src.exists():
            print(f"  [skip] {name}: {src} not found")
            continue

        dst = SCRIPTS / spec.script
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not force:
            print(f"  [keep] {spec.script} (already vendored; --force to replace)")
            continue

        text = src.read_text()
        if spec.runtime == Runtime.R:
            # GENECI's wrappers source the helper via a container-absolute path.
            text = text.replace(
                'source("components/infer_network/functions.R")',
                'source(file.path(dirname(sys.frame(1)$ofile %||% "."), "functions.R"))'
                if False else
                'local({\n'
                '  args <- commandArgs(trailingOnly = FALSE)\n'
                '  self <- sub("^--file=", "", args[grep("^--file=", args)])\n'
                '  source(file.path(dirname(self), "functions.R"))\n'
                '}, envir = globalenv())',
            )
        dst.write_text(ATTRIBUTION + text)
        print(f"  {name:<14} -> {dst.relative_to(HERE.parent.parent)}")

    # MEOMI ships an entire src/ tree rather than a single script.
    meomi_src = src_root / "MEOMI" / "src"
    if meomi_src.is_dir():
        dst = SCRIPTS / "r" / "MEOMI_src"
        if dst.exists() and force:
            shutil.rmtree(dst)
        if not dst.exists():
            shutil.copytree(meomi_src, dst)
            print(f"  MEOMI src/     -> {dst.relative_to(HERE.parent.parent)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geneci-path", required=True, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite scripts that are already vendored")
    args = ap.parse_args()
    print("Vendoring GENECI wrapper scripts into MultiGCN ...")
    vendor(args.geneci_path, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
