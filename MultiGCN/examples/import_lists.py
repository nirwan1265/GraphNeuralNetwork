#!/usr/bin/env python
"""
Import previously-computed edge lists into a MultiGCN output folder.

Inference is expensive and its results are just files. If an algorithm already
ran - under GENECI, on another machine, before a Docker image broke - there is
no reason to recompute it. This copies GRN_<ALGO>.csv files into a run's
lists/ directory, after checking they are actually usable.

This is how the GENECI-container algorithms (MEOMI, NONLINEARODES, and the
MATLAB family) stay available: the images no longer start on modern kernels,
but their historical output is perfectly good.

    python examples/import_lists.py \\
        --from ../data/inferred_networks/dream4_100_03_exp/lists \\
        --output dream4_net3_out

    python examples/import_lists.py --from <dir> --only MEOMI TIGRESS
    python examples/import_lists.py --from <dir> --overwrite
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from multigcn.registry import ALGORITHMS


def inspect(path: Path):
    """Return (n_edges, n_distinct_scores, problem_or_None)."""
    try:
        df = pd.read_csv(path, header=None, names=["source", "target", "score"])
    except Exception as exc:
        return 0, 0, f"unreadable: {exc}"
    if df.empty:
        return 0, 0, "empty file"
    if df["score"].isna().any():
        return len(df), 0, "contains NaN scores"

    distinct = df["score"].nunique()
    # The pre-binarised trap: a "confidence" column with two values has no
    # gradation left, so every downstream ranking step is meaningless.
    if distinct <= 2:
        return len(df), distinct, (
            f"only {distinct} distinct score(s) - looks pre-binarised, "
            "not continuous")
    return len(df), distinct, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", required=True, type=Path,
                    help="Directory containing GRN_<ALGO>.csv files")
    ap.add_argument("--output", type=Path, default=Path("dream4_net3_out"))
    ap.add_argument("--only", nargs="+", default=None,
                    help="Import just these algorithms")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace lists already present in the destination")
    ap.add_argument("--allow-unregistered", action="store_true",
                    help="Import files whose name is not a known algorithm")
    args = ap.parse_args()

    src = args.source.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"Not a directory: {src}")

    dst = (args.output / "lists").resolve()
    dst.mkdir(parents=True, exist_ok=True)

    found = sorted(src.glob("GRN_*.csv"))
    if not found:
        raise SystemExit(f"No GRN_*.csv files in {src}")

    print(f"Source: {src}")
    print(f"Target: {dst}\n")

    imported, skipped, rejected = [], [], []
    for path in found:
        name = path.stem[len("GRN_"):]

        if args.only and name not in args.only:
            continue
        if name not in ALGORITHMS and not args.allow_unregistered:
            rejected.append((name, "not a registered algorithm "
                                   "(use --allow-unregistered to force)"))
            continue

        target = dst / path.name
        if target.exists() and not args.overwrite:
            skipped.append((name, "already present (--overwrite to replace)"))
            continue

        n_edges, distinct, problem = inspect(path)
        if problem:
            rejected.append((name, problem))
            continue

        shutil.copyfile(path, target)
        imported.append((name, n_edges, distinct))

    for name, n_edges, distinct in imported:
        print(f"  imported  {name:<16} {n_edges:>8} edges, "
              f"{distinct} distinct scores")
    for name, why in skipped:
        print(f"  skipped   {name:<16} {why}")
    for name, why in rejected:
        print(f"  REJECTED  {name:<16} {why}")

    print(f"\n{len(imported)} imported, {len(skipped)} skipped, "
          f"{len(rejected)} rejected.")

    present = sorted(p.stem[len("GRN_"):] for p in dst.glob("GRN_*.csv"))
    print(f"\n{len(present)} algorithm(s) now available in {dst}:")
    print("  " + ", ".join(present))


if __name__ == "__main__":
    main()
