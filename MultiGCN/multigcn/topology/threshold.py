"""Rank-based input topology construction."""
from __future__ import annotations

import numpy as np


def top_x_percent_edges(matrix: np.ndarray, top_x: float = 0.10,
                        keep_weights: bool = False,
                        symmetric: bool = True) -> np.ndarray:
    """
    Keep each algorithm's own top `top_x` fraction of gene pairs as edges.

    Rank-based rather than an absolute score cutoff, because the algorithms
    produce scores on entirely different internal scales - at a fixed cutoff of
    0.6, LEAP retained ~67% of all pairs while others retained under 0.3%.

    symmetric=True (the validated default) ranks only the UPPER TRIANGLE and
    mirrors each retained pair, so the graph handed to the GCN is undirected.
    This matters: it means top_x is a fraction of the n(n-1)/2 unordered pairs,
    not of all n^2 ordered ones, and the two settings do NOT produce the same
    edge count. Directed input was not what the architecture was validated on.

    top_x = 0.10 is the validated default.
    """
    if not 0 < top_x <= 1:
        raise ValueError(f"top_x must be in (0, 1], got {top_x}")

    n = matrix.shape[0]

    if not symmetric:
        flat = matrix.flatten()
        k = max(1, int(round(top_x * flat.size)))
        idx = np.argpartition(flat, -k)[-k:]
        mask = np.zeros(flat.size, dtype=bool)
        mask[idx] = True
        mask &= flat > 0
        mask = mask.reshape(n, n)
        return np.where(mask, matrix, 0.0).astype(np.float32) if keep_weights \
            else mask.astype(np.float32)

    triu = np.triu_indices(n, k=1)
    scores = matrix[triu]

    # Rank only pairs that actually carry a score. Taking the top k by argsort
    # alone pads sparse algorithms with zero-score ties: C3NET emits nonzero
    # values for ~1.6% of pairs, so a flat top-10% would fill the remaining 8%
    # with arbitrary picks among equal zeros and report a density it never
    # produced.
    nonzero = scores > 0
    k = min(max(1, int(len(scores) * top_x)), int(nonzero.sum()))
    if k == 0:
        raise ValueError("Matrix has no positive scores; cannot build a topology.")

    candidate_idx = np.flatnonzero(nonzero)
    order = np.argsort(scores[candidate_idx])[::-1][:k]
    top_idx = candidate_idx[order]

    out = np.zeros((n, n), dtype=np.float32)
    rows, cols = triu[0][top_idx], triu[1][top_idx]
    values = matrix[rows, cols] if keep_weights else 1.0
    out[rows, cols] = values
    out[cols, rows] = values
    return out


def normalized_laplacian_edge_index(adj: np.ndarray):
    """
    Build the edge_index the GCN is trained on.

    The validated pipeline does NOT feed the raw adjacency to GCNConv. It adds
    self-loops, then forms the symmetric normalised Laplacian

        L = I - D^-1/2 (A + I) D^-1/2

    and takes that sparsity pattern as the edge index. Keep this: the encoder
    weights were tuned against this input, and swapping in the plain adjacency
    changes what the first layer sees.
    """
    import scipy.sparse as sp
    import torch
    from torch_geometric.utils import from_scipy_sparse_matrix

    a = sp.csr_matrix(adj)
    a = a + sp.eye(a.shape[0])

    coo = sp.coo_matrix(a)
    rowsum = np.array(coo.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-5, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat = sp.diags(d_inv_sqrt)
    laplacian = sp.eye(coo.shape[0]) - d_mat.dot(coo).dot(d_mat)

    edge_index, _ = from_scipy_sparse_matrix(laplacian.tocoo())
    return edge_index


def density(adj: np.ndarray) -> float:
    n = adj.shape[0]
    return float((adj > 0).sum()) / (n * n)


def edge_index_from_adj(adj: np.ndarray):
    """Plain dense adjacency -> edge_index, without Laplacian normalisation."""
    import torch
    rows, cols = np.nonzero(adj)
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
