"""
MultiGCN wrapper for dynGENIE3.

dynGENIE3 models the rate of change of expression, so it REQUIRES time-series
data. Steady-state data can be supplied in addition (SS_data) but never
instead - there is no dY/dt to regress against without time points.

The dynGENIE3 source is not on PyPI. Point MultiGCN at a local clone with
    dyngenie3_source_path=/path/to/dynGENIE3/dynGENIE3_python
or set the DYNGENIE3_PATH environment variable.

Usage: DYNGENIE3.py <ss_file_or_-> <out_dir> <ts_file> [params_json]

Time-series CSV format: first column 'time', remaining columns are genes.
Multiple experiments are separated by the time value resetting downward.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_dyngenie3(explicit=None):
    # Try the explicit setting, the environment variable, then the handful of
    # places a clone realistically lives. Requiring a flag for something the
    # code can find on its own is a papercut that recurs on every run.
    candidates = [explicit, os.environ.get("DYNGENIE3_PATH")]
    for base in ["~/Documents/Github/dynGENIE3", "~/Github/dynGENIE3",
                 "~/dynGENIE3"]:
        candidates += [base + "/dynGENIE3_python", base]

    for candidate in filter(None, candidates):
        p = Path(candidate).expanduser()
        if (p / "dynGENIE3.py").exists():
            sys.path.insert(0, str(p))
            break
    try:
        from dynGENIE3 import dynGENIE3, get_link_list  # noqa
    except ImportError:
        sys.exit(
            "Could not import dynGENIE3. It is not a PyPI package - clone it "
            "from github.com/vahuynh/dynGENIE3 and pass\n"
            "  --dyngenie3-source-path /path/to/dynGENIE3/dynGENIE3_python\n"
            "or set DYNGENIE3_PATH."
        )
    return dynGENIE3, get_link_list


def split_experiments(df, time_col):
    """A decrease in the time value marks the start of a new time series."""
    times = df[time_col].to_numpy()
    breaks = [0] + [i for i in range(1, len(times)) if times[i] <= times[i - 1]]
    breaks.append(len(times))
    return [(slice(breaks[i], breaks[i + 1])) for i in range(len(breaks) - 1)]


def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: DYNGENIE3.py <ss_file|-> <out_dir> <ts_file> [params_json]")
    ss_file, out_dir, ts_file = sys.argv[1], sys.argv[2], sys.argv[3]
    params = json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else {}

    dynGENIE3, _ = load_dyngenie3(params.get("source_path"))

    raw = pd.read_csv(ts_file)
    tcol = next((c for c in raw.columns if c.lower() == "time"), None)
    if tcol is None:
        sys.exit(f"{ts_file} has no 'time' column. dynGENIE3 requires time series.")

    gene_names = [c for c in raw.columns if c != tcol]
    TS_data, time_points = [], []
    for sl in split_experiments(raw, tcol):
        chunk = raw.iloc[sl]
        TS_data.append(chunk[gene_names].to_numpy(dtype=float))
        time_points.append(chunk[tcol].to_numpy(dtype=float))
    print(f"dynGENIE3: {len(TS_data)} time series, {len(gene_names)} genes")

    SS_data = None
    if ss_file and ss_file != "-" and Path(ss_file).exists():
        SS_data = pd.read_csv(ss_file, index_col=0).T
        SS_data = SS_data[gene_names].to_numpy(dtype=float)
        print(f"  plus {SS_data.shape[0]} steady-state conditions")

    alpha = params.get("alpha", "from_data")
    if alpha != "from_data":
        alpha = float(alpha)
    K = params.get("K", "sqrt")
    if K not in ("sqrt", "all"):
        K = int(K)

    VIM = dynGENIE3(
        TS_data, time_points, alpha=alpha, SS_data=SS_data,
        gene_names=gene_names, tree_method=params.get("tree_method", "RF"),
        K=K, ntrees=int(params.get("ntrees", 1000)),
        nthreads=int(params.get("nthreads", 1)),
    )[0]

    rows = [(gene_names[i], gene_names[j], float(VIM[i, j]))
            for i in range(len(gene_names)) for j in range(len(gene_names))
            if i != j and VIM[i, j] > 0]
    df = pd.DataFrame(rows, columns=["source", "target", "score"])
    span = df["score"].max() - df["score"].min()
    df["score"] = 0.0 if span == 0 else (df["score"] - df["score"].min()) / span
    df = df[df["score"] != 0].sort_values("score", ascending=False)

    out = Path(out_dir) / "GRN_DYNGENIE3.csv"
    df.to_csv(out, header=False, index=False)
    print(f"wrote {len(df)} edges to {out}")


if __name__ == "__main__":
    main()
