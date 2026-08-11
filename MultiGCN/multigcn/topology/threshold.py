"""Rank-based input topology construction."""
from __future__ import annotations

import numpy as np


def top_x_percent_edges(matrix: np.ndarray, top_x: float = 0.10,
                        keep_weights: bool = False) -> np.ndarray:
    """
    Keep each algorithm's own top `top_x` fraction of gene pairs as edges.

    Rank-based rather than an absolute score cutoff, because the 16 algorithms
    produce scores on entirely different internal scales - at a fixed cutoff of
    0.6, LEAP retained ~67% of all pairs while others retained under 0.3%.

    top_x = 0.10 is the validated default: roughly 2-3x the true regulatory
    density of the DREAM4 networks (3.5-5%). For a new organism, estimate that
    organism's expected density independently (e.g. RegulonDB TF-target
    statistics for E. coli) and apply the same multiplier. Never tune top_x
    against the ground truth of the dataset being analysed.
    """
    if not 0 < top_x <= 1:
        raise ValueError(f"top_x must be in (0, 1], got {top_x}")

    n = matrix.shape[0]
    flat = matrix.flatten()
    k = max(1, int(round(top_x * flat.size)))

    cutoff_idx = np.argpartition(flat, -k)[-k:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[cutoff_idx] = True
    mask &= flat > 0  # never promote a zero-score pair into an edge
    mask = mask.reshape(n, n)

    if keep_weights:
        return np.where(mask, matrix, 0.0).astype(np.float32)
    return mask.astype(np.float32)


def density(adj: np.ndarray) -> float:
    n = adj.shape[0]
    return float((adj > 0).sum()) / (n * n)


def edge_index_from_adj(adj: np.ndarray):
    """Dense adjacency -> torch_geometric edge_index (2, num_edges)."""
    import torch
    rows, cols = np.nonzero(adj)
    return torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
