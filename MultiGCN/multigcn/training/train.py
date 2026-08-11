"""GAE training loop."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..models.encoders import GCNEncoder, identity_features
from ..models.decoders import build_decoder


def negative_sample(edge_index: torch.Tensor, num_nodes: int,
                    num_neg: int, device) -> torch.Tensor:
    """
    Uniform random negative sampling from non-edges.

    Deliberately uniform. Model-score-based hard-negative mining failed
    catastrophically (near-random on every network) because it punishes the
    encoder for confidently proposing genuine hidden links - which is exactly
    the behaviour the method depends on. Structural (common-neighbour) mining
    did not generalise either.
    """
    src = torch.randint(0, num_nodes, (num_neg,), device=device)
    dst = torch.randint(0, num_nodes, (num_neg,), device=device)
    return torch.stack([src, dst], dim=0)


def train_gae(
    edge_index: torch.Tensor,
    num_nodes: int,
    decoder: str = "bilinear",
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 1e-8,
    hidden: int = 32,
    out_channels: int = 16,
    dropout: float = 0.01,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Train one GAE and return its dense reconstruction as a numpy array.

    Defaults are the validated production settings. 300 epochs (not the
    original 150) because every decoder variant tested showed real further
    gains past 150.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if seed is not None:
        torch.manual_seed(seed)

    edge_index = edge_index.to(device)
    x = identity_features(num_nodes, device=device)

    encoder = GCNEncoder(num_nodes, hidden, out_channels, dropout).to(device)
    dec = build_decoder(decoder, out_channels).to(device)

    params = list(encoder.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    num_pos = edge_index.size(1)
    encoder.train(); dec.train()

    for epoch in range(epochs):
        opt.zero_grad()
        z = encoder(x, edge_index)

        pos_logits = dec(z, edge_index)
        neg_index = negative_sample(edge_index, num_nodes, num_pos, device)
        neg_logits = dec(z, neg_index)

        loss = (
            F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits))
            + F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits))
        )
        loss.backward()
        opt.step()

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch+1:4d}/{epochs}  loss {loss.item():.4f}")

    encoder.eval(); dec.eval()
    with torch.no_grad():
        z = encoder(x, edge_index)
        recon = torch.sigmoid(dec(z)).cpu().numpy()
    return recon.astype(np.float32)
