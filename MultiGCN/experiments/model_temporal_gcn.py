#!/usr/bin/env python
"""
Model 2 of 3: temporal GCN + GRU over time-varying node features.

Naming
------
This is NOT EvolveGCN in the strict sense. EvolveGCN evolves the GCN *weight
matrices* through an RNN across a sequence of graph snapshots. Here each
algorithm emits one static network, so there are no snapshots to evolve over.
What varies with time is the node features. The correct family is GCRN/STGCN:
static topology, temporal features, GCN per timestep, recurrence across
timesteps.

Why this and not more weighting
-------------------------------
Two weighting approaches have now failed. Learned consensus weights degraded
fusion on all three DREAM4 networks (0.1154/0.0839/0.1460 vs uniform
0.1428/0.1210/0.1846) and showed no reliable relationship with per-algorithm
accuracy (r = -0.18, -0.00, -0.44 across nets 1-3; the one significant value
did not replicate). Reshuffling weights over the same inputs is not where the
leverage is.

The real asymmetry is informational. Production node features are
identity_features(n) - one-hot gene IDs - so the GCN never sees expression data
at all, let alone the time axis. SWING wins DREAM4 by 2-4x because it is the
only method using temporal structure; the other 20 algorithms receive the 210
columns as 210 independent conditions and discard the ordering. This model
gives the GCN that same temporal information.

Data layout
-----------
DREAM4 expression columns are named 'perturbation.<series>.t<time>': 10 series
x 21 timepoints = 210 samples. Parsed here directly with a regex rather than
through multigcn.data.dream4, so the script is self-contained and fails loudly
on any other format.

For each timepoint t we build X_t (n_genes x n_series), run a GCN over the
algorithm's static graph, and feed the resulting node embeddings through a
GRUCell across t = 1..21 (nodes act as the batch dimension). The final hidden
state is the embedding; the decoder and loss are unchanged, so the only
difference from production is what the encoder is allowed to see.

--mode static is the control: same expression values as features, but all 210
columns at once and no recurrence. It separates "expression helps" from
"temporal ORDER helps", which is the actual question. Reporting temporal
without this control would not distinguish the two.

Usage
-----
    cd ~/Documents/Github/GraphNeuralNetwork/MultiGCN
    D=~/Documents/Github/GraphNeuralNetwork/data/Database.DREAM4
    python experiments/model_temporal_gcn.py \
        --lists dream4_net3_out/lists \
        --expression $D/EXP/dream4_100_03_exp.csv \
        --gold $D/GS/dream4_100_03_gs.csv \
        --top-x 0.08 --seeds 0 1 2 --mode temporal
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variants import (build_graph, evaluation_mask, masked_metrics, minmax,
                      raw_consensus)

from multigcn.data.loaders import load_edge_list, read_expression
from multigcn.evaluation.metrics import load_gold_standard
from multigcn.models.decoders import build_decoder

from torch_geometric.nn import GAE, GCNConv

# DREAM4 columns: 'perturbation.<series>.t0', then
# 'perturbation.<series>.applied.t50'..'.t500', then
# 'perturbation.<series>.removed.t550'..'.t1000'. 21 evenly spaced points per
# series, with a regime switch at t=500 when the perturbation is lifted. The
# middle segment is a PHASE LABEL, not part of the series identity - treating
# 'applied' and 'removed' as separate series would cut every course in half.
COL_RE = re.compile(
    r"^perturbation\.(\d+)\.(?:(applied|removed)\.)?t(\d+)$", re.IGNORECASE)


def build_time_tensor(expr: pd.DataFrame):
    """
    -> (T, n_genes, n_series) float32 tensor, list of times.

    Raises if the columns are not in DREAM4 time-series form. Silently falling
    back to treating columns as unordered would produce a plausible-looking
    number that means nothing, so this fails loudly instead.
    """
    parsed = {}
    for c in expr.columns:
        m = COL_RE.match(str(c))
        if m:
            # group(2) is the phase label and is deliberately ignored: time
            # already orders the points across both phases.
            parsed[c] = (int(m.group(1)), int(m.group(3)))
    if len(parsed) != len(expr.columns):
        bad = [c for c in expr.columns if c not in parsed][:4]
        raise SystemExit(
            f"Expression columns are not DREAM4 time-series format "
            f"('perturbation.<n>[.applied|.removed].t<time>'). Unparsed "
            f"examples: {bad}. This model needs ordered timepoints; DREAM5 "
            f"needs the separate prep_dream5_timeseries.py path and is not "
            f"supported here."
        )

    series = sorted({s for s, _ in parsed.values()})
    times = sorted({t for _, t in parsed.values()})
    lookup = {(s, t): c for c, (s, t) in parsed.items()}
    missing = [(s, t) for s in series for t in times if (s, t) not in lookup]
    if missing:
        raise SystemExit(f"Ragged design: {len(missing)} (series, time) cells "
                         f"absent, e.g. {missing[:3]}. A GRU over timepoints "
                         f"needs every series measured at every time.")

    vals = expr.to_numpy(dtype=np.float32)
    # Standardise per gene across all samples so genes are comparable; done
    # once globally rather than per timepoint, which would erase the temporal
    # signal we are trying to feed the model.
    mu = vals.mean(axis=1, keepdims=True)
    sd = vals.std(axis=1, keepdims=True) + 1e-8
    z = (vals - mu) / sd
    pos = {c: i for i, c in enumerate(expr.columns)}

    X = np.stack([
        np.stack([z[:, pos[lookup[(s, t)]]] for s in series], axis=1)
        for t in times
    ])
    return torch.from_numpy(X), times


class TemporalEncoder(torch.nn.Module):
    """
    Per timestep: two GCN layers over the static graph. Across timesteps: a
    GRUCell whose batch dimension is the nodes. Final hidden state -> embedding.
    """

    def __init__(self, in_channels, hidden=16, out_channels=16, dropout=0.01):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * hidden, cached=False)
        self.conv2 = GCNConv(2 * hidden, 2 * hidden, cached=False)
        self.bn1 = torch.nn.BatchNorm1d(2 * hidden)
        self.gru = torch.nn.GRUCell(2 * hidden, 2 * hidden)
        self.out = torch.nn.Linear(2 * hidden, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.hidden2 = 2 * hidden

    def forward(self, X_seq, edge_index, edge_weight=None):
        n = X_seq.shape[1]
        h = torch.zeros(n, self.hidden2, device=X_seq.device)
        for t in range(X_seq.shape[0]):
            g = F.relu(self.conv1(X_seq[t], edge_index, edge_weight))
            g = F.relu(self.bn1(self.conv2(g, edge_index, edge_weight)))
            h = self.gru(self.dropout(g), h)
        return F.relu(self.out(h))


class StaticEncoder(torch.nn.Module):
    """Control: identical depth, expression as features, no recurrence."""

    def __init__(self, in_channels, hidden=16, out_channels=16, dropout=0.01):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * hidden, cached=False)
        self.conv2 = GCNConv(2 * hidden, 2 * hidden, cached=False)
        self.conv3 = GCNConv(2 * hidden, out_channels, cached=False)
        self.bn1 = torch.nn.BatchNorm1d(2 * hidden)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, X, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(X, edge_index, edge_weight))
        x2 = F.relu(self.bn1(self.conv2(x1, edge_index, edge_weight)))
        x2 = self.dropout(x2) + x1
        return F.relu(self.conv3(x2, edge_index, edge_weight))


def train_one(mode, feats, edge_index, edge_weight, decoder, epochs, lr, wd,
              hidden, out_ch, dropout, seed, device):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dev = torch.device(device or "cpu")
    feats = feats.to(dev)
    edge_index = edge_index.to(dev)
    if edge_weight is not None:
        edge_weight = edge_weight.to(dev)

    in_ch = feats.shape[-1]
    enc = (TemporalEncoder(in_ch, hidden, out_ch, dropout) if mode == "temporal"
           else StaticEncoder(in_ch, hidden, out_ch, dropout))
    model = GAE(enc, build_decoder(decoder, out_ch)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        z = model.encoder(feats, edge_index, edge_weight)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        z = model.encoder(feats, edge_index, edge_weight)
        return minmax(model.decoder.forward_all(z).cpu().numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lists", required=True, type=Path)
    p.add_argument("--expression", required=True, type=Path)
    p.add_argument("--gold", required=True, type=Path)
    p.add_argument("--top-x", type=float, default=0.08)
    p.add_argument("--mode", choices=["temporal", "static"], default="temporal")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--decoders", nargs="+", default=["bilinear", "distance"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path,
                   default=Path("experiments/temporal_gcn.csv"))
    a = p.parse_args()

    expr = read_expression(a.expression)
    genes = list(expr.index)
    truth = load_gold_standard(a.gold, genes)
    n = len(genes)
    mask = evaluation_mask(n, undirected=True)

    def score(m):
        return masked_metrics(m, truth, mask)

    X_seq, times = build_time_tensor(expr)
    print(f"{n} genes, {X_seq.shape[0]} timepoints x {X_seq.shape[2]} series "
          f"(t={times[0]}..{times[-1]})")
    feats = X_seq if a.mode == "temporal" else torch.from_numpy(
        ((expr.to_numpy(dtype=np.float32)
          - expr.to_numpy(dtype=np.float32).mean(1, keepdims=True))
         / (expr.to_numpy(dtype=np.float32).std(1, keepdims=True) + 1e-8)))
    print(f"mode={a.mode}, feature dim {feats.shape[-1]}")

    matrices = {}
    for path in sorted(a.lists.glob("GRN_*.csv")):
        try:
            m = load_edge_list(path, genes)
        except Exception:
            continue
        if len(np.unique(m[m > 0])) < 2:
            continue
        matrices[path.stem[4:]] = m
    names = sorted(matrices)
    print(f"{len(names)} algorithms")

    per = {k: score(matrices[k])["aupr"] for k in names}
    best = max(per, key=per.get)
    print(f"best single input: {best} AUPR {per[best]:.4f}   |   "
          f"mean {np.mean(list(per.values())):.4f}")
    r0 = score(raw_consensus([matrices[k] for k in names]))
    print(f"CONTROL raw rank-average, no GCN: AUPR {r0['aupr']:.4f}\n")

    graphs = {k: build_graph(matrices[k], a.top_x, weighted=True)
              for k in names}

    rows = []
    for seed in a.seeds:
        t0 = time.time()
        recons = []
        for k in names:
            ei, ew = graphs[k]
            for dec in a.decoders:
                recons.append(train_one(a.mode, feats, ei, ew, dec, a.epochs,
                                        a.lr, a.weight_decay, a.hidden,
                                        a.embedding_dim, a.dropout, seed,
                                        a.device))
        fused = np.mean(np.stack(recons), axis=0)
        r = score(fused)
        rows.append({"seed": seed, "mode": a.mode,
                     "auroc": r["auroc"], "aupr": r["aupr"]})
        print(f"  seed={seed}  AUROC {r['auroc']:.4f}  AUPR {r['aupr']:.4f}  "
              f"({len(recons)} recons, {time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    print(f"\n{a.mode}: AUPR {df.aupr.mean():.4f} +/- {df.aupr.std():.4f}   "
          f"AUROC {df.auroc.mean():.4f} +/- {df.auroc.std():.4f}")
    print(f"control (no GCN)  {r0['aupr']:.4f}")
    print(f"best single input {best} {per[best]:.4f}")
    print("\nCompare against identity-feature uniform fusion for this network "
          "(net1 0.1428, net2 0.1210, net3 0.1846) and run --mode static to "
          "separate 'expression helps' from 'temporal order helps'.")


if __name__ == "__main__":
    main()
