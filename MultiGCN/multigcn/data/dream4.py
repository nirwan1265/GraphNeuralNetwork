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


def prepare_insilico(folder, out_dir=None, include=("wildtype", "knockdowns",
                                                     "timeseries")) -> dict:
    """
    Turn one DREAM4 *In Silico Size 100* network folder (as downloaded from
    Synapse: *_wildtype.tsv, *_knockouts.tsv, *_knockdowns.tsv,
    *_timeseries.tsv) into MultiGCN's three inputs:

        expression.csv   genes x samples: every non-knockout sample in
                         `include` (steady-state view, as PORTIA uses it)
        timeseries.csv   'time' + genes, experiments concatenated
        knockouts.tsv    the knockout experiments, row k = knockout of gene k

    Knockouts are kept out of expression.csv on purpose: they are passed
    separately (infer(knockouts=...)) so the algorithms that use them know
    which gene was knocked out.

    Returns {"expression": path, "time_series": path, "knockouts": path}.
    """
    import shutil
    folder = Path(folder)
    out_dir = Path(out_dir or folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    def find(kind):
        hits = sorted(folder.glob(f"*_{kind}.tsv"))
        if not hits:
            raise FileNotFoundError(f"no *_{kind}.tsv in {folder}")
        return hits[0]

    blocks = []
    for kind in include:
        df = pd.read_csv(find(kind), sep="\t")
        df.columns = [str(c).strip('"') for c in df.columns]
        if kind == "timeseries":
            df = df.drop(columns=[c for c in df.columns if c.lower() == "time"])
        df = df.dropna(how="all")
        df.index = [f"{kind}.{k + 1}" for k in range(len(df))]
        blocks.append(df)
    expr = pd.concat(blocks).T
    paths = {"expression": out_dir / "expression.csv",
             "time_series": out_dir / "timeseries.csv",
             "knockouts": out_dir / "knockouts.tsv"}
    expr.to_csv(paths["expression"])

    ts = pd.read_csv(find("timeseries"), sep="\t").dropna(how="all")
    ts.columns = [str(c).strip('"') for c in ts.columns]
    ts = ts.rename(columns={c: "time" for c in ts.columns if c.lower() == "time"})
    ts.to_csv(paths["time_series"], index=False)

    shutil.copyfile(find("knockouts"), paths["knockouts"])
    return paths
