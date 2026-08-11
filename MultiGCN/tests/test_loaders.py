import numpy as np
import pandas as pd
import pytest

from multigcn.data.loaders import DataError, load_algorithm_matrices, load_edge_list


def test_load_edge_list_shape(lists_dir, genes):
    m = load_edge_list(lists_dir / "GRN_CLR.csv", genes)
    assert m.shape == (len(genes), len(genes))
    assert m.min() >= 0


def test_rejects_prebinarised_input(tmp_path, genes):
    """The normalized_weight_matrix trap: binary values masquerading as scores."""
    rows = [(genes[i], genes[j], 1.0)
            for i in range(len(genes)) for j in range(len(genes)) if i != j]
    p = tmp_path / "GRN_FAKE.csv"
    pd.DataFrame(rows).to_csv(p, header=False, index=False)
    with pytest.raises(DataError, match="pre-binarised"):
        load_edge_list(p, genes)


def test_missing_method_fails_loudly(lists_dir, genes):
    with pytest.raises(DataError, match="no edge list"):
        load_algorithm_matrices(lists_dir, genes, methods=["CLR", "TIGRESS"])
