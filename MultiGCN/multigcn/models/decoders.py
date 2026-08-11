"""Decoders. The validated production setting is the bilinear+distance ensemble."""
from __future__ import annotations

import torch
import torch.nn as nn


class InnerProductDecoder(nn.Module):
    """
    Baseline: sigmoid(z_i . z_j).

    Structurally symmetric, so it cannot represent direction - a TF regulating
    a target is not the same relation reversed. Kept for comparison only.
    """
    def forward(self, z, edge_index=None):
        logits = z @ z.t()
        return logits if edge_index is None else logits[edge_index[0], edge_index[1]]


class BilinearDecoder(nn.Module):
    """
    sigmoid(z_i^T W z_j) with learned W, initialised near identity.

    Because W is unconstrained, z_i^T W z_j != z_j^T W z_i in general, so this
    can represent directed regulation. Beats inner-product on AUPR consistently.
    """
    def __init__(self, dim: int = 16, init_noise: float = 0.01):
        super().__init__()
        w = torch.eye(dim) + init_noise * torch.randn(dim, dim)
        self.W = nn.Parameter(w)

    def forward(self, z, edge_index=None):
        logits = z @ self.W @ z.t()
        return logits if edge_index is None else logits[edge_index[0], edge_index[1]]


class DistanceDecoder(nn.Module):
    """
    sigmoid(-||z_i - z_j||^2). Proximity in embedding space rather than a
    learned interaction matrix - a genuinely different inductive bias, which is
    why ensembling it with the bilinear decoder helps.
    """
    def forward(self, z, edge_index=None):
        if edge_index is None:
            sq = (z * z).sum(dim=1, keepdim=True)
            logits = -(sq + sq.t() - 2 * (z @ z.t()))
            return logits
        diff = z[edge_index[0]] - z[edge_index[1]]
        return -(diff * diff).sum(dim=1)


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
