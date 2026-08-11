"""GCN encoder. Architecture is validated - see docs before changing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "MultiGCN needs torch-geometric. Install it matching your torch/CUDA "
        "build: https://pytorch-geometric.readthedocs.io/en/latest/install/"
    ) from e


class GCNEncoder(nn.Module):
    """
    3-layer GCN, widths (32, 32, 16), residual connection on layer 2.

    Validated against: 2-layer, 4-layer (32,32,16,8), GraphSAGE, GAT,
    Jumping Knowledge, and four alternative width shapes (funnel, wide-middle,
    bottleneck, constant-wide). None beat this configuration. Do not change
    without re-running the full DREAM4 comparison.
    """

    def __init__(self, num_nodes: int, hidden: int = 32,
                 out_channels: int = 16, dropout: float = 0.01):
        super().__init__()
        self.conv1 = GCNConv(num_nodes, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = GCNConv(hidden, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))

        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h2 = h2 + h1  # residual

        return F.relu(self.conv3(h2, edge_index))


def identity_features(num_nodes: int, device=None) -> torch.Tensor:
    """
    Node features = identity matrix (one-hot gene id, no expression values).

    Tested directly against using real expression values as node features;
    identity won clearly on every metric. Identity forces the encoder to learn
    from graph topology alone, which is what the reconstruction objective
    actually rewards.
    """
    return torch.eye(num_nodes, device=device)
