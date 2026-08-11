"""
DREAM4 input conversion.

The DREAM4 *_exp.csv files are genes x samples, where the samples are not
independent conditions: they are 10 perturbation experiments of 21 ordered
time points each (t0 .. t1000 in steps of 50), laid end to end.

Steady-state algorithms can consume that matrix as-is - treating the 210
columns as 210 conditions - which is what GENECI does and what the existing
DREAM4 results were produced with. Time-series algorithms need the temporal
structure made explicit, which is what this module does.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

COLUMN_RE = re.compile(
    r"^perturbation\.(?P<exp>\d+)\.(?:(?P<phase>applied|removed)\.)?t(?P<time>\d+)$"
)


def parse_columns(columns) -> List[Tuple[int, int, str]]:
    """-> [(experiment, time, original_column_name)] in file order."""
    parsed = []
    for c in columns:
        m = COLUMN_RE.match(str(c))
        if m is None:
            raise ValueError(
                f"Column {c!r} does not look like DREAM4 time-series data. "
                "Expected 'perturbation.<n>.[applied|removed.]t<time>'."
            )
        parsed.append((int(m.group("exp")), int(m.group("time")), c))
    return parsed


def is_time_series(path) -> bool:
    """True if the file's columns encode perturbation time courses."""
    try:
        cols = pd.read_csv(path, index_col=0, nrows=0).columns
        parse_columns(cols)
        return True
    except Exception:
        return False


def to_time_series(exp_path, out_path: Optional[Path] = None) -> Path:
    """
    Convert a DREAM4 *_exp.csv into the format MultiGCN's time-series wrappers
    expect: first column 'time', remaining columns genes, one row per
    measurement, experiments concatenated. A time value that does not increase
    marks the start of the next experiment.
    """
    exp_path = Path(exp_path)
    df = pd.read_csv(exp_path, index_col=0)
    parsed = parse_columns(df.columns)

    # Sort by (experiment, time) so concatenated blocks are properly ordered
    # even if the file is not already in that order.
    order = sorted(range(len(parsed)), key=lambda i: (parsed[i][0], parsed[i][1]))

    genes = list(df.index)
    rows, times = [], []
    for i in order:
        _exp, t, col = parsed[i]
        rows.append(df[col].to_numpy())
        times.append(t)

    out = pd.DataFrame(rows, columns=genes)
    out.insert(0, "time", times)

    if out_path is None:
        out_path = exp_path.with_name(exp_path.stem + "_timeseries.csv")
    out_path = Path(out_path)
    out.to_csv(out_path, index=False)
    return out_path


def summarise(exp_path) -> str:
    df = pd.read_csv(exp_path, index_col=0)
    parsed = parse_columns(df.columns)
    exps = sorted({e for e, _, _ in parsed})
    times = sorted({t for _, t, _ in parsed})
    return (f"{len(df)} genes x {df.shape[1]} samples = "
            f"{len(exps)} experiments x {len(times)} time points "
            f"(t{times[0]}..t{times[-1]})")
