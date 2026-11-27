"""GCN encoder. Architecture is validated - see docs/REJECTED.md before changing."""
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
    3-layer GCN, widths (2*h, 2*h, out) = (32, 32, 16) at the default h=16.

    Layer 2 is BatchNorm'd, dropped out, then residually added to layer 1's
    output. Note the order: dropout is applied BEFORE the residual add, so the
    skip connection carries an undropped path. That is what was validated.

    Validated against: 2-layer, 4-layer (32,32,16,8), GraphSAGE, GAT, Jumping
    Knowledge, and four alternative width shapes. None beat this. Do not change
    without re-running the full DREAM4 comparison.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 16,
                 out_channels: int = 16, dropout_rate: float = 0.01):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * hidden_channels, cached=False)
        self.conv2 = GCNConv(2 * hidden_channels, 2 * hidden_channels, cached=False)
        self.conv3 = GCNConv(2 * hidden_channels, out_channels, cached=False)
        self.bn1 = nn.BatchNorm1d(2 * hidden_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.bn1(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2) + x1
        return F.relu(self.conv3(x2, edge_index))


def identity_features(num_nodes: int, device=None) -> torch.Tensor:
    """
    Node features = identity matrix (one-hot gene id, no expression values).

    Tested directly against using real expression values as node features;
    identity won clearly on every metric. Identity forces the encoder to learn
    from graph topology alone, which is what the reconstruction objective
    actually rewards.
    """
    return torch.eye(num_nodes, device=device)
