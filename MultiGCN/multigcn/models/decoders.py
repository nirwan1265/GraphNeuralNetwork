"""
Decoders. The validated production setting is the bilinear+distance ensemble.

Each decoder exposes the interface torch_geometric's GAE expects:
    forward(z, edge_index, sigmoid=True)  -> score per edge
    forward_all(z, sigmoid=True)          -> dense score matrix
so they can be dropped straight into GAE(encoder, decoder).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InnerProductDecoder(nn.Module):
    """
    Baseline: sigmoid(z_i . z_j).

    Structurally symmetric, so it cannot represent direction - a TF regulating
    a target is not the same relation reversed. Kept for comparison only.
    """

    def forward(self, z, edge_index, sigmoid=True):
        src, dst = edge_index
        value = (z[src] * z[dst]).sum(dim=-1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        value = z @ z.t()
        return torch.sigmoid(value) if sigmoid else value


class BilinearDecoder(nn.Module):
    """
    sigmoid(z_i^T W z_j), W learned and initialised near identity.

    Because W is unconstrained, z_i^T W z_j != z_j^T W z_i in general, so this
    can represent directed regulation. Best single decoder in testing.
    """

    def __init__(self, embed_dim: int = 16, init_noise: float = 0.01):
        super().__init__()
        self.W = nn.Parameter(
            torch.eye(embed_dim) + init_noise * torch.randn(embed_dim, embed_dim))

    def forward(self, z, edge_index, sigmoid=True):
        src, dst = edge_index
        value = torch.sum((z[src] @ self.W) * z[dst], dim=-1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        value = (z @ self.W) @ z.t()
        return torch.sigmoid(value) if sigmoid else value


class DistanceDecoder(nn.Module):
    """
    sigmoid(-||z_i - z_j||^2). Proximity in embedding space rather than a
    learned interaction matrix - a genuinely different inductive bias, which is
    why ensembling it with the bilinear decoder helps.
    """

    def forward(self, z, edge_index, sigmoid=True):
        src, dst = edge_index
        value = -((z[src] - z[dst]) ** 2).sum(dim=-1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        value = -(torch.cdist(z, z) ** 2)
        return torch.sigmoid(value) if sigmoid else value


DECODERS = {
    "inner_product": InnerProductDecoder,
    "bilinear": BilinearDecoder,
    "distance": DistanceDecoder,
}

DEFAULT_DECODERS = ["bilinear", "distance"]


def build_decoder(name: str, dim: int = 16) -> nn.Module:
    if name not in DECODERS:
        raise ValueError(f"Unknown decoder {name!r}. Available: {list(DECODERS)}")
    cls = DECODERS[name]
    return cls(dim) if cls is BilinearDecoder else cls()
