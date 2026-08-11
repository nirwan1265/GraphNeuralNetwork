import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def genes():
    return [f"G{i}" for i in range(20)]


@pytest.fixture
def expression(tmp_path, genes):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(len(genes), 15)),
                      index=genes, columns=[f"S{j}" for j in range(15)])
    p = tmp_path / "expression.csv"
    df.to_csv(p)
    return p


@pytest.fixture
def lists_dir(tmp_path, genes):
    """Two fake algorithm outputs with continuous scores."""
    d = tmp_path / "lists"
    d.mkdir()
    rng = np.random.default_rng(1)
    for algo in ["CLR", "ARACNE"]:
        rows = []
        for i, s in enumerate(genes):
            for j, t in enumerate(genes):
                if i != j and rng.random() < 0.4:
                    rows.append((s, t, float(rng.random())))
        pd.DataFrame(rows).to_csv(d / f"GRN_{algo}.csv", header=False, index=False)
    return d
