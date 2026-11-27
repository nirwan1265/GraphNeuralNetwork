"""GAE training loop."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..models.decoders import build_decoder
from ..models.encoders import GCNEncoder, identity_features


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_gae(
    edge_index: torch.Tensor,
    num_nodes: int,
    decoder: str = "bilinear",
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 1e-8,
    hidden: int = 16,
    out_channels: int = 16,
    dropout: float = 0.01,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    x: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Train one GAE and return its dense reconstruction, min-max scaled to [0,1].

    Uses torch_geometric's GAE.recon_loss, which scores the observed edges as
    positives and draws an equal number of uniform random non-edges as
    negatives each epoch. Deliberately uniform: model-score-based hard-negative
    mining failed catastrophically, because it punishes the encoder for
    confidently proposing genuine hidden links - exactly the behaviour the
    method depends on.

    Defaults are the validated production settings. 300 epochs (not 150)
    because every decoder variant tested still improved past 150.
    """
    from torch_geometric.nn import GAE

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if seed is not None:
        set_seed(seed)

    edge_index = edge_index.to(device)
    if x is None:
        x = identity_features(num_nodes, device=device)
    else:
        x = x.to(device)

    encoder = GCNEncoder(num_nodes, hidden, out_channels, dropout)
    model = GAE(encoder, build_decoder(decoder, out_channels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch + 1:4d}/{epochs}  loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
        recon = model.decoder.forward_all(z).cpu().numpy()

    span = recon.max() - recon.min()
    return ((recon - recon.min()) / (span + 1e-12)).astype(np.float32)
