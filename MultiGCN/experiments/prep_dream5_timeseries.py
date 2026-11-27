#!/usr/bin/env python
"""
Build a MultiGCN time-series view from DREAM5 metadata.

Why this cannot reuse multigcn/data/dream4.py
---------------------------------------------
dream4.py parses temporal structure out of *column names*
('perturbation.3.t150') in a rigid format: 10 experiments x 21 evenly spaced
points, nothing else in the file. DREAM5 differs on every count:

  * timing lives in a separate chip_features.tsv, row-aligned to the
    expression matrix;
  * steady-state rows (Time == NA) are interleaved with timed rows and must be
    dropped from the time-series view;
  * one #Experiment can contain several distinct courses that differ only by
    perturbation and/or which genes were deleted or overexpressed - Network1
    experiment 105 holds at least four separate 5-point courses;
  * spacing is uneven (experiment 107: 0, 40, 240 minutes);
  * replicates recur at the same timepoint (experiment 106 has Repeat 1 and 2
    at every point).

A course is therefore identified by
(#Experiment, Perturbations, DeletedGenes, OverexpressedGenes, Repeat), not by
#Experiment alone. Collapsing all timed rows into one sequence would splice
unrelated perturbations end to end and hand the time-series algorithms noise.

Caveat worth checking before trusting results: DREAM5 courses average ~3.2
points (vs 21 in DREAM4). Lag-based methods such as SWING have very little
window to work with at that length. --min-points drops the shortest courses;
report how many survive.

Usage
-----
    python experiments/prep_dream5_timeseries.py \
        --network-dir "$DR/Network3" \
        --expression ~/.../dream5_net3_prepped/expression.csv \
        --out ~/.../dream5_net3_prepped/expression_timeseries.csv \
        --min-points 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COURSE_KEYS = ["#Experiment", "Perturbations", "DeletedGenes",
               "OverexpressedGenes", "Repeat"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network-dir", required=True, type=str,
                   help="DREAM5 NetworkN directory (holds 'input data/')")
    p.add_argument("--expression", required=True, type=str,
                   help="genes x samples CSV from prep_dream5.py")
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--min-points", type=int, default=3,
                   help="Drop courses with fewer than this many timepoints")
    p.add_argument("--uniform-length", type=int, default=None,
                   help="Emit a BALANCED design: keep only courses with at "
                        "least N timepoints, truncate each to the first N, "
                        "and relabel time as ordinal position 1..N. Required "
                        "by SWING, which groups rows by time VALUE and fails "
                        "on ragged grids.")
    p.add_argument("--collapse-replicates", action="store_true",
                   help="Average samples sharing a course and timepoint "
                        "instead of treating each replicate as its own course")
    args = p.parse_args()

    root = Path(args.network_dir).expanduser()
    feat_path = next((root / "input data").glob("*chip_features.tsv"))
    feats = pd.read_csv(feat_path, sep="\t", dtype=str)
    feats.columns = [c.strip() for c in feats.columns]

    expr = pd.read_csv(Path(args.expression).expanduser(), index_col=0)
    genes = list(expr.index)
    print(f"expression: {len(genes)} genes x {expr.shape[1]} samples")
    print(f"features:   {len(feats)} rows from {feat_path.name}")

    if len(feats) != expr.shape[1]:
        raise SystemExit(
            f"Row/sample mismatch: {len(feats)} feature rows vs "
            f"{expr.shape[1]} expression samples. chip_features must be "
            f"row-aligned to the expression columns; check that "
            f"prep_dream5.py did not reorder or drop samples."
        )

    for k in COURSE_KEYS + ["Time"]:
        if k not in feats.columns:
            raise SystemExit(f"chip_features is missing column {k!r}. "
                             f"Found: {list(feats.columns)}")

    # Position in the expression matrix, before any filtering.
    feats = feats.reset_index(drop=True)
    feats["_pos"] = np.arange(len(feats))

    timed = feats[feats["Time"].notna() & (feats["Time"] != "NA")].copy()
    timed["Time"] = timed["Time"].astype(float)
    print(f"timed rows:  {len(timed)}  "
          f"(steady-state rows dropped: {len(feats) - len(timed)})")

    if args.collapse_replicates:
        keys = [k for k in COURSE_KEYS if k != "Repeat"]
    else:
        keys = COURSE_KEYS
    timed[keys] = timed[keys].fillna("NA")

    groups = list(timed.groupby(keys, sort=False, dropna=False))
    lengths = pd.Series([g["Time"].nunique() for _, g in groups])
    print(f"courses:     {len(groups)}   points per course: "
          f"min {lengths.min()}, median {int(lengths.median())}, "
          f"max {lengths.max()}, mean {lengths.mean():.1f}")
    print("  length distribution: "
          + ", ".join(f"{n}pt x{c}" for n, c in
                      sorted(lengths.value_counts().items())))

    kept, dropped = [], 0
    threshold = args.uniform_length or args.min_points
    for key, g in groups:
        if g["Time"].nunique() < threshold:
            dropped += 1
            continue
        kept.append((key, g.sort_values("Time")))

    if not kept:
        raise SystemExit(
            f"No course has >= {threshold} timepoints. Lower the threshold, "
            f"or accept that this network has no usable time-series "
            f"structure for lag-based methods."
        )
    print(f"kept {len(kept)} courses (dropped {dropped} shorter than "
          f"{threshold} points)")
    if args.uniform_length:
        print(f"balanced design: truncating each to the first "
              f"{args.uniform_length} points, time relabelled 1.."
              f"{args.uniform_length}")

    # Emit the format the wrappers expect: 'time' then one column per gene,
    # courses concatenated. A non-increasing time marks a new course, which is
    # the same convention dream4.py uses.
    blocks = []
    for _key, g in kept:
        if args.collapse_replicates:
            rows = []
            for t, sub in g.groupby("Time", sort=True):
                vals = expr.iloc[:, sub["_pos"].to_numpy()].mean(axis=1)
                rows.append((t, vals.to_numpy()))
        else:
            rows = [(float(r.Time), expr.iloc[:, int(r._pos)].to_numpy())
                    for r in g.itertuples()]
        blk = pd.DataFrame([v for _t, v in rows], columns=genes)
        blk.insert(0, "time", [t for t, _v in rows])
        if args.uniform_length:
            # Truncate to the first N points and replace real times with
            # ordinal positions, so every course lands on an identical grid.
            blk = blk.iloc[:args.uniform_length].copy()
            blk["time"] = np.arange(1, len(blk) + 1)
        blocks.append(blk)

    out_df = pd.concat(blocks, ignore_index=True)
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n-> {out_path}  ({out_df.shape[0]} rows x {len(genes)} genes)")

    if lengths.mean() < 5:
        print("\nWARNING: mean course length is short. SWING and other "
              "lag-based methods may extract little signal here; treat any "
              "poor result as a property of the data, not of the method.")


if __name__ == "__main__":
    main()
