"""
Joint multi-relational graph autoencoder: one model over every algorithm.

Each algorithm's graph is one edge type in a single multigraph on the genes.
An R-GCN encoder (Schlichtkrull et al. 2018) gives every gene ONE embedding
that has to explain all the graphs at once; a DistMult decoder per algorithm
reads that shared embedding in its own way:

    s_r(i, j) = sigmoid( sum_k  z_ik * R_rk * z_jk )

Why this and not one GAE per algorithm: a separate GAE only ever sees its own
algorithm's graph and is trained to rebuild that same graph, so it can do
little more than smooth it (it ties with the closed-form Katz index). Sharing
the gene embeddings across algorithms forces the model to keep the structure
the algorithms agree on, which is where the predicted new edges come from.

Basis decomposition (num_bases) makes the relations share weights. It is
essential: without it each relation gets its own weights, the model is back
to separate graphs, and DREAM4 AUPR fell from 0.164 to 0.107.

DistMult is symmetric, so this model scores undirected pairs only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import RGCNConv
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MultiGCN needs torch-geometric. Install it matching your torch/CUDA "
        "build: https://pytorch-geometric.readthedocs.io/en/latest/install/"
    ) from e


class JointEncoder(nn.Module):
    """Two R-GCN layers, n -> 2*dim -> dim, ReLU + dropout in between."""

    def __init__(self, num_nodes: int, num_relations: int, dim: int = 16,
                 num_bases: int = 8, dropout: float = 0.01):
        super().__init__()
        nb = num_bases if num_bases and num_bases < num_relations else None
        self.conv1 = RGCNConv(num_nodes, 2 * dim, num_relations, num_bases=nb)
        self.conv2 = RGCNConv(2 * dim, dim, num_relations, num_bases=nb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        """x=None means one-hot identity features: RGCNConv then indexes its
        weights directly instead of multiplying by an n x n identity matrix -
        the same model, but O(n) instead of O(n^2) work in the first layer."""
        x = self.dropout(F.relu(self.conv1(x, edge_index, edge_type)))
        return self.conv2(x, edge_index, edge_type)


class DistMultDecoder(nn.Module):
    """One diagonal relation vector per algorithm, initialised near ones."""

    def __init__(self, num_relations: int, dim: int = 16, init_noise: float = 0.01):
        super().__init__()
        self.rel = nn.Parameter(torch.ones(num_relations, dim)
                                + init_noise * torch.randn(num_relations, dim))

    def forward(self, z, edge_index, relation: int):
        """Logits for the given pairs under one relation."""
        return (z[edge_index[0]] * self.rel[relation] * z[edge_index[1]]).sum(-1)

    def forward_all(self, z):
        """Probabilities for every pair under every relation: (R, n, n)."""
        return torch.sigmoid(torch.einsum("ik,rk,jk->rij", z, self.rel, z))

    def forward_relation(self, z, relation: int):
        """Probabilities for every pair under one relation: (n, n)."""
        return torch.sigmoid((z * self.rel[relation]) @ z.t())
