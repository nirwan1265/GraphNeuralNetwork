"""GAE training loop."""
from __future__ import annotations

from typing import Dict, Optional

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


def train_joint(
    adjacencies: Dict[str, np.ndarray],
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 1e-8,
    dim: int = 16,
    num_bases: int = 8,
    dropout: float = 0.01,
    negatives: int = 1,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Train ONE multi-relational GAE over every algorithm's graph.

    adjacencies[name] is that algorithm's symmetric 0/1 (or weighted, >0)
    input topology; each becomes one edge type. Per epoch and per algorithm,
    its edges are positives and `negatives` x as many uniform random pairs are
    negatives (binary cross-entropy), averaged over algorithms.

    Returns {name: dense reconstruction under that algorithm's decoder,
    min-max scaled to [0, 1]}.

    Defaults are the DREAM4-validated settings. Longer training and larger
    embeddings lower the loss but lower AUPR too (1200 epochs / dim 64: loss
    0.44, AUPR 0.151 vs 0.166) - a model that can rebuild every graph exactly
    has memorised them instead of keeping what they share.
    """
    from ..models.relational import DistMultDecoder, JointEncoder

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if seed is not None:
        set_seed(seed)

    names = list(adjacencies)
    n = next(iter(adjacencies.values())).shape[0]
    pos, src, dst, typ = [], [], [], []
    for r, name in enumerate(names):
        A = adjacencies[name]
        if hasattr(A, "tocoo"):                       # scipy.sparse
            A = A.tocoo(); keep = A.col > A.row
            i, j = A.row[keep], A.col[keep]
        else:
            i, j = np.nonzero(np.triu(A, 1))
        if len(i) == 0:
            raise ValueError(f"{name}: empty input topology")
        pos.append(torch.tensor(np.vstack([i, j]), dtype=torch.long, device=device))
        src += [i, j]; dst += [j, i]; typ.append(np.full(2 * len(i), r))
    edge_index = torch.tensor(np.vstack([np.concatenate(src), np.concatenate(dst)]),
                              dtype=torch.long, device=device)
    edge_type = torch.tensor(np.concatenate(typ), dtype=torch.long, device=device)

    R = len(names)
    encoder = JointEncoder(n, R, dim, num_bases, dropout).to(device)
    decoder = DistMultDecoder(R, dim).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    x = None          # one-hot identity, handled inside RGCNConv without an n x n matrix
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    for epoch in range(epochs):
        encoder.train(); opt.zero_grad()
        z = encoder(x, edge_index, edge_type)
        loss = 0.0
        for r in range(R):
            p = pos[r]
            neg = torch.randint(0, n, (2, negatives * p.shape[1]), device=device)
            lp, ln = decoder(z, p, r), decoder(z, neg, r)
            loss = loss + bce(lp, torch.ones_like(lp)) + bce(ln, torch.zeros_like(ln))
        loss = loss / R
        loss.backward(); opt.step()
        if verbose and (epoch + 1) % 50 == 0:
            print(f"    epoch {epoch + 1:4d}/{epochs}  loss {loss.item():.4f}")

    encoder.eval()
    out = {}
    with torch.no_grad():
        z = encoder(x, edge_index, edge_type)
        for r, name in enumerate(names):          # one n x n matrix at a time
            s = decoder.forward_relation(z, r).cpu().numpy().astype(np.float32)
            lo, hi = float(s.min()), float(s.max())
            s -= lo; s /= (hi - lo + 1e-12)
            out[name] = s
    return out
