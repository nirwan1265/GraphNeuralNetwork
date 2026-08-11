"""
MultiGCN wrapper for SWING (sliding-window Granger causality).

SWING is the only method in the pipeline that tests Granger causality: does
gene A's *past* improve prediction of gene B's *future*, beyond B's own past?
That is a different question from mutual information (association) or tree
importance (predictive power at the same time point), which is exactly why it
is worth including.

TIME SERIES REQUIRED. SWING slices the series into overlapping windows, fits a
regression per window at several lags, then aggregates edge importances across
windows and lags.

SWING is not on PyPI. Point MultiGCN at a local clone with
    swing_source_path=/path/to/SWING
or set the SWING_PATH environment variable.

Compatibility note
------------------
SWING's requirements.txt pins numpy 1.11.3 / pandas 0.19.2 / scikit-learn
0.19.0 (2016). Those pins are not load-bearing: setup.py declares the same
dependencies unpinned, and the only removed API the source actually uses is
DataFrame.append (dropped in pandas 2.0). We restore that one method below
rather than editing the SWING clone, so the upstream checkout stays pristine
and `git pull` keeps working.

Usage: SWING.py <ss_file|-> <out_dir> <ts_file> [params_json]
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


def install_pandas_compat_shim():
    """Restore DataFrame.append for libraries written before pandas 2.0."""
    if not hasattr(pd.DataFrame, "append"):
        def _append(self, other, ignore_index=False, sort=False, **kwargs):
            other = other if isinstance(other, (list, tuple)) else [other]
            return pd.concat([self, *other], ignore_index=ignore_index, sort=sort)
        pd.DataFrame.append = _append
        return True
    return False


def load_swing(explicit=None):
    for candidate in filter(None, [explicit, os.environ.get("SWING_PATH")]):
        p = Path(candidate).expanduser()
        if (p / "Swing").is_dir():
            sys.path.insert(0, str(p))
            break
    try:
        from Swing import Swing
    except ImportError:
        sys.exit(
            "Could not import Swing. It is not a PyPI package - clone it from "
            "github.com/bagherilab/SWING and pass\n"
            "  swing_source_path=/path/to/SWING\n"
            "or set SWING_PATH."
        )
    return Swing


def to_swing_tsv(ts_file: Path) -> tuple:
    """
    SWING reads a TSV from disk itself, with the time column first. Convert
    whatever we were given into that shape and return (path, time_label).
    """
    raw = pd.read_csv(ts_file)
    tcol = next((c for c in raw.columns if c.lower() == "time"), None)
    if tcol is None:
        sys.exit(f"{ts_file} has no 'time' column. SWING requires time series.")

    genes = [c for c in raw.columns if c != tcol]
    ordered = raw[[tcol] + genes]

    tmp = Path(tempfile.mkdtemp()) / "swing_input.tsv"
    ordered.to_csv(tmp, sep="\t", index=False)
    return tmp, tcol, genes


def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: SWING.py <ss_file|-> <out_dir> <ts_file> [params_json]")
    out_dir, ts_file = sys.argv[2], sys.argv[3]
    params = json.loads(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else {}

    if install_pandas_compat_shim():
        print("SWING: applied DataFrame.append compatibility shim")

    Swing = load_swing(params.get("source_path") or None)
    tsv_path, time_label, genes = to_swing_tsv(Path(ts_file))
    print(f"SWING: {len(genes)} genes, window={params.get('window_width', 10)}, "
          f"lags {params.get('min_lag', 1)}-{params.get('max_lag', 3)}")

    sg = Swing(
        str(tsv_path),
        1,                      # gene_start: column 0 is time
        None,                   # gene_end
        time_label,
        "\t",
        min_lag=int(params.get("min_lag", 1)),
        max_lag=int(params.get("max_lag", 3)),
        window_width=int(params.get("window_width", 10)),
        window_type=params.get("window_type", "RandomForest"),
    )

    sg.zscore_all_data()
    sg.create_windows()
    sg.optimize_params()
    sg.fit_windows(n_trees=int(params.get("n_trees", 500)),
                   show_progress=False,
                   n_jobs=int(params.get("n_jobs", -1)))
    sg.compile_edges(self_edges=False)
    sg.make_static_edge_dict(self_edges=False, lag_method="mean_mean")

    sort_by = params.get("sort_by", "mean")
    ranked = sg.make_sort_df(sg.edge_dict, sort_by=sort_by)

    # 'regulator-target' holds (source, target) tuples; the score column is
    # named after whichever statistic was requested.
    score_col = f"{sort_by}_importance"
    if score_col not in ranked.columns:
        sys.exit(f"SWING returned no '{score_col}' column; got {list(ranked.columns)}")

    df = pd.DataFrame({
        "source": [e[0] for e in ranked["regulator-target"]],
        "target": [e[1] for e in ranked["regulator-target"]],
        "score": ranked[score_col].astype(float).values,
    })
    df = df[df["source"] != df["target"]].dropna()

    span = df["score"].max() - df["score"].min()
    df["score"] = 0.0 if span == 0 else (df["score"] - df["score"].min()) / span
    df = df[df["score"] != 0].sort_values("score", ascending=False)

    out = Path(out_dir) / "GRN_SWING.csv"
    df.to_csv(out, header=False, index=False)
    print(f"wrote {len(df)} edges to {out}")


if __name__ == "__main__":
    main()
