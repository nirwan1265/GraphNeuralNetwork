#!/usr/bin/env python
"""
Ablation harness for the MultiGCN fusion step. Standalone - imports the
package but changes nothing in it.

Diagnosis this is testing
-------------------------
In the production pipeline the only thing that reaches the GCN, per algorithm,
is *which pairs are edges*:

  * Config.keep_weights defaults to False, so retained edges become 1.0; and
  * normalized_laplacian_edge_index() keeps only the sparsity pattern of the
    Laplacian, discarding the values from_scipy_sparse_matrix() returns.

So keep_weights=True is a silent no-op whenever laplacian=True, and SWING
(AUPR 0.18) and PPCOR (AUPR 0.05) both arrive as binary graphs of identical
density. The model cannot tell them apart, and a flat mean over the
reconstructions lands near the mean input rather than the best one.

Axes
----
  topology   binary        production: unweighted edges, Laplacian pattern
             weighted      rank-normalised edge weights, passed to GCNConv

  fusion     minmax_mean   production: min-max each recon, average
             rank_mean     quantile-rank each recon, average
             blend_<a>     a * rank_mean(recons) + (1-a) * rank consensus of
                           the RAW inputs. a=0.0 is a no-GCN control.

  top_x      swept
  seed       replicated - everything is reported as mean +/- sd over seeds

Why seeds matter here
---------------------
A single-seed sweep of this grid produced a spread of 0.118-0.132 AUPR across
almost every configuration, with one outlier at 0.1485 that sat at a
non-monotonic kink in the top_x curve. Differences that size are not
distinguishable without replication, so nothing in this harness reports a
single run.

RAW_RANK_MEAN is the control that matters: a rank-average of the raw inputs
with no GCN anywhere. If the GCN configurations do not beat it, the model is
not contributing.

Usage
-----
    cd ~/Documents/Github/GraphNeuralNetwork/MultiGCN
    python experiments/variants.py \
        --lists dream4_net2_out/lists \
        --expression "$EXP" --gold "$GS" \
        --top-x 0.05 0.10 0.20 --seeds 0 1 2 3 4 \
        --alphas 0.0 0.25 0.5 0.75 1.0 \
        --out experiments/results_net2.csv
"""
from __future__ import annotations

import argparse
import itertools
import os
import random
import sys
import time
from pathlib import Path

# Must be set before torch initialises its CUDA/CPU kernels, hence before the
# torch import below rather than inside train_one().
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

from multigcn.data.loaders import load_edge_list, read_expression
from multigcn.evaluation.metrics import load_gold_standard
from multigcn.models.decoders import build_decoder
from multigcn.models.encoders import identity_features
from multigcn.topology.threshold import top_x_percent_edges

from sklearn.metrics import average_precision_score, roc_auc_score

from torch_geometric.nn import GAE, GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix


# ------------------------------------------------------------------- scoring

def evaluation_mask(n: int, tf_idx=None, undirected: bool = True):
    """
    Boolean mask of the gene pairs an evaluation is entitled to score.

    DREAM4 assessed every pair, so the mask is the upper triangle (undirected)
    or everything off-diagonal (directed).

    DREAM5 only ever assessed TF -> target pairs: the gold standard lists ~195
    transcription factors as sources and says nothing about the other ~1450
    genes as regulators. Scoring all pairs silently treats every unassessed
    pair as a true negative, which deflates AUPR by roughly an order of
    magnitude and makes the numbers incomparable to published DREAM5 results.
    Passing tf_idx restricts scoring to TF rows, which is what the challenge
    actually evaluated.
    """
    if tf_idx is not None:
        mask = np.zeros((n, n), dtype=bool)
        mask[np.asarray(tf_idx), :] = True
        np.fill_diagonal(mask, False)
        return mask
    if undirected:
        return np.triu(np.ones((n, n), dtype=bool), k=1)
    return ~np.eye(n, dtype=bool)


def masked_metrics(scores: np.ndarray, truth: np.ndarray,
                   mask: np.ndarray) -> dict:
    """AUROC/AUPR over the masked pairs only."""
    y = (truth[mask] > 0).astype(np.int8)
    s = scores[mask].astype(np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return {"auroc": float("nan"), "aupr": float("nan")}
    return {"auroc": float(roc_auc_score(y, s)),
            "aupr": float(average_precision_score(y, s))}


# ---------------------------------------------------------------- normalising

def rank_normalise(mat: np.ndarray) -> np.ndarray:
    """
    Map every entry to its quantile rank in [0,1].

    Min-max is outlier-sensitive: one extreme value compresses everything else
    toward zero, so a heavy-tailed reconstruction contributes almost nothing to
    a mean of min-max'd matrices. Ranks make each matrix contribute its
    *ordering*, which is what AUROC/AUPR score, on a common scale.

    Not "Borda count on binarised networks" (rejected in docs/REJECTED.md) -
    nothing is binarised and the full ordering survives.
    """
    flat = np.asarray(mat, dtype=np.float64).ravel()
    r = rankdata(flat, method="average")
    r = (r - 1.0) / max(len(r) - 1, 1)
    return r.reshape(mat.shape).astype(np.float32)


def minmax(mat: np.ndarray) -> np.ndarray:
    lo, hi = float(mat.min()), float(mat.max())
    if hi - lo < 1e-12:
        return np.zeros_like(mat, dtype=np.float32)
    return ((mat - lo) / (hi - lo)).astype(np.float32)


# ------------------------------------------------------------------- topology

def build_graph(mat: np.ndarray, top_x: float, weighted: bool):
    """
    Return (edge_index, edge_weight | None) for one algorithm's matrix.

    binary   : exactly the production path - Laplacian sparsity pattern, no
               weights anywhere.
    weighted : keep each retained edge's score, rank-normalise it *within this
               algorithm* so scales are comparable across algorithms, and use
               D^-1/2 (A+I) D^-1/2 as edge weights.

    The weighted branch uses A+I normalised rather than
    L = I - D^-1/2 (A+I) D^-1/2 on purpose: L's off-diagonals are negative, and
    a negative edge weight into GCNConv means "subtract this neighbour", which
    is not what a confidence score means. The binary branch is untouched, so
    the production configuration is still reproduced exactly.
    """
    adj = top_x_percent_edges(mat, top_x, keep_weights=weighted, symmetric=True)

    if not weighted:
        a = sp.csr_matrix((adj > 0).astype(np.float32)) + sp.eye(adj.shape[0])
        coo = sp.coo_matrix(a)
        rowsum = np.asarray(coo.sum(1)).ravel()
        d_inv = np.power(rowsum + 1e-5, -0.5)
        d_inv[np.isinf(d_inv)] = 0.0
        d = sp.diags(d_inv)
        lap = sp.eye(coo.shape[0]) - d.dot(coo).dot(d)
        edge_index, _ = from_scipy_sparse_matrix(lap.tocoo())
        return edge_index, None

    nz = adj > 0
    if nz.sum() == 0:
        raise ValueError("no edges retained")
    w = np.zeros_like(adj, dtype=np.float32)
    vals = adj[nz]
    # Rank within this algorithm, floored at 0.1 so the weakest retained edge
    # still passes some signal rather than being switched off entirely.
    w[nz] = ((rankdata(vals, method="average") - 1.0)
             / max(len(vals) - 1, 1)).astype(np.float32) * 0.9 + 0.1

    a = sp.csr_matrix(w) + sp.eye(adj.shape[0], dtype=np.float32)
    coo = sp.coo_matrix(a)
    rowsum = np.asarray(coo.sum(1)).ravel()
    d_inv = np.power(rowsum + 1e-5, -0.5)
    d_inv[np.isinf(d_inv)] = 0.0
    d = sp.diags(d_inv)
    norm = sp.coo_matrix(d.dot(coo).dot(d))
    edge_index, edge_weight = from_scipy_sparse_matrix(norm)
    return edge_index, edge_weight.float()


# -------------------------------------------------------------------- encoder

class WeightedGCNEncoder(torch.nn.Module):
    """
    Same shape as multigcn.models.encoders.GCNEncoder - 3 layers, (2h, 2h, out),
    BatchNorm on layer 2, dropout before the residual add - but forwards
    edge_weight to each GCNConv. With edge_weight=None it is behaviourally
    identical to the production encoder.
    """

    def __init__(self, in_channels, hidden=16, out_channels=16, dropout=0.01):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * hidden, cached=False)
        self.conv2 = GCNConv(2 * hidden, 2 * hidden, cached=False)
        self.conv3 = GCNConv(2 * hidden, out_channels, cached=False)
        self.bn1 = torch.nn.BatchNorm1d(2 * hidden)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x2 = F.relu(self.bn1(self.conv2(x1, edge_index, edge_weight)))
        x2 = self.dropout(x2) + x1
        return F.relu(self.conv3(x2, edge_index, edge_weight))


def train_one(edge_index, edge_weight, n, decoder, epochs, lr, weight_decay,
              hidden, out_channels, dropout, seed, device):
    """
    One GAE, returned as a dense reconstruction. Mirrors train_gae().

    Determinism: seeding torch/numpy alone is not enough. PyG's message passing
    uses scatter-add, whose floating-point summation order is not fixed, so the
    same seed can give materially different results - at n=1643 the same config
    produced AUPR 0.0959 and 0.0356 on two runs. use_deterministic_algorithms
    forces deterministic kernels where they exist; warn_only=True means ops
    with no deterministic implementation warn instead of raising, so this stays
    usable while still removing the main source of drift. Determinism costs
    some speed, which is the right trade for numbers going into a paper.
    """
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    edge_index = edge_index.to(dev)
    if edge_weight is not None:
        edge_weight = edge_weight.to(dev)
    x = identity_features(n, device=dev)

    model = GAE(WeightedGCNEncoder(n, hidden, out_channels, dropout),
                build_decoder(decoder, out_channels)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        z = model.encoder(x, edge_index, edge_weight)
        # Weighting changes message passing, not what counts as a positive, so
        # this matches the production loss.
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        z = model.encoder(x, edge_index, edge_weight)
        recon = model.decoder.forward_all(z).cpu().numpy()
    return minmax(recon)


# --------------------------------------------------------------------- fusion

def fuse_minmax(recons):
    return np.mean(np.stack([minmax(m) for m in recons]), axis=0).astype(np.float32)


def fuse_rank(recons):
    return np.mean(np.stack([rank_normalise(m) for m in recons]), axis=0).astype(np.float32)


def raw_consensus(raw_inputs):
    """Rank-average of the raw algorithm outputs. No GCN. Seed-independent."""
    return np.mean(np.stack([rank_normalise(m) for m in raw_inputs]),
                   axis=0).astype(np.float32)


def fuse_blend(gcn_rank, raw_rank, alpha):
    """
    alpha * reconstruction consensus + (1-alpha) * raw consensus.

    An autoencoder trained to reconstruct its input can at best return it, so
    for the *best* input the round-trip through a 16-dim bottleneck is pure
    loss - and production fuse() never sees the raw scores at all. alpha=0 is
    the no-GCN control; alpha=1 is plain rank_mean.
    """
    return (alpha * gcn_rank + (1 - alpha) * raw_rank).astype(np.float32)


# ----------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lists", required=True, type=Path)
    p.add_argument("--expression", required=True, type=Path)
    p.add_argument("--gold", required=True, type=Path)
    p.add_argument("--tfs", type=Path, default=None,
                   help="One gene id per line. If given, evaluation is "
                        "restricted to TF x gene pairs (DREAM5 convention) "
                        "and scored directionally.")
    p.add_argument("--top-x", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    p.add_argument("--topologies", nargs="+", default=["binary", "weighted"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                   help="Replicates. Everything is reported as mean +/- sd.")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="Blend weights on the GCN consensus. 0.0 = no GCN.")
    p.add_argument("--decoders", nargs="+", default=["bilinear", "distance"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=Path("experiments/results.csv"))
    args = p.parse_args()

    genes = list(read_expression(args.expression).index)
    truth = load_gold_standard(args.gold, genes)
    n = len(genes)

    if args.tfs:
        wanted = [ln.strip() for ln in open(args.tfs) if ln.strip()]
        pos = {g: i for i, g in enumerate(genes)}
        tf_idx = [pos[g] for g in wanted if g in pos]
        missing = len(wanted) - len(tf_idx)
        mask = evaluation_mask(n, tf_idx=tf_idx)
        print(f"evaluating TF x gene pairs only: {len(tf_idx)} TFs"
              + (f" ({missing} not in expression)" if missing else "")
              + f", {mask.sum():,} scored pairs")
    else:
        mask = evaluation_mask(n, undirected=True)
        print(f"evaluating all gene pairs (upper triangle): "
              f"{mask.sum():,} scored pairs")

    n_pos = int((truth[mask] > 0).sum())
    print(f"positives in scored region: {n_pos:,} "
          f"({100*n_pos/mask.sum():.3f}%)")

    def score(m):
        return masked_metrics(m, truth, mask)

    matrices, rejected = {}, {}
    for path in sorted(args.lists.glob("GRN_*.csv")):
        name = path.stem[4:]
        try:
            m = load_edge_list(path, genes)
        except Exception as exc:
            rejected[name] = str(exc).split(". ")[0]
            continue
        if len(np.unique(m[m > 0])) < 2:
            rejected[name] = "constant scores"
            continue
        matrices[name] = m

    if not matrices:
        sys.exit(f"No usable edge lists in {args.lists}")
    print(f"{len(matrices)} algorithms: {', '.join(sorted(matrices))}")
    if rejected:
        print("excluded: " + ", ".join(f"{k} ({v})" for k, v in rejected.items()))

    inputs = []
    for name, m in matrices.items():
        r = score(m)
        inputs.append({"method": name, "auroc": r["auroc"], "aupr": r["aupr"]})
    inputs.sort(key=lambda r: -r["aupr"])
    best_in, mean_in = inputs[0], float(np.mean([r["aupr"] for r in inputs]))
    print(f"best single input: {best_in['method']} AUPR {best_in['aupr']:.4f} "
          f"AUROC {best_in['auroc']:.4f}   |   mean input AUPR {mean_in:.4f}")

    raw_rank = raw_consensus(list(matrices.values()))
    r0 = score(raw_rank)
    print(f"CONTROL raw rank-average, no GCN: AUROC {r0['auroc']:.4f}  "
          f"AUPR {r0['aupr']:.4f}\n")

    rows = []
    for top_x, topo, seed in itertools.product(args.top_x, args.topologies,
                                               args.seeds):
        t0 = time.time()
        recons, failed = [], []
        for name, m in matrices.items():
            try:
                ei, ew = build_graph(m, top_x, weighted=(topo == "weighted"))
            except Exception as exc:
                failed.append(f"{name}: {exc}")
                continue
            for dec in args.decoders:
                recons.append(train_one(
                    ei, ew, n, dec, args.epochs, args.lr, args.weight_decay,
                    args.hidden, args.embedding_dim, args.dropout,
                    seed, args.device))
        if not recons:
            print(f"top_x={top_x} {topo} seed={seed}: all failed; skipping")
            continue

        gcn_rank = fuse_rank(recons)
        results = {"minmax_mean": fuse_minmax(recons), "rank_mean": gcn_rank}
        for a in args.alphas:
            results[f"blend_{a:g}"] = fuse_blend(gcn_rank, raw_rank, a)

        for fusion, fused in results.items():
            r = score(fused)
            rows.append({"topology": topo, "fusion": fusion, "top_x": top_x,
                         "seed": seed, "auroc": r["auroc"], "aupr": r["aupr"]})
        print(f"  top_x={top_x:<5} {topo:<8} seed={seed}  "
              f"({len(recons)} recons, {time.time()-t0:.0f}s)"
              + (f"  failed: {'; '.join(failed)}" if failed else ""))

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    agg = (df.groupby(["topology", "fusion", "top_x"])
             .agg(aupr_mean=("aupr", "mean"), aupr_sd=("aupr", "std"),
                  auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
                  n=("aupr", "size"))
             .reset_index()
             .sort_values("aupr_mean", ascending=False))
    agg.to_csv(args.out.with_name(args.out.stem + "_summary.csv"), index=False)

    print(f"\n-> {args.out}")
    print("\nAUPR, mean +/- sd over seeds (top 15):\n")
    print(f"{'topology':<9} {'fusion':<13} {'top_x':>6}   {'AUPR':>16}   {'AUROC':>16}")
    for _, r in agg.head(15).iterrows():
        print(f"{r.topology:<9} {r.fusion:<13} {r.top_x:>6.3f}   "
              f"{r.aupr_mean:.4f} +/- {r.aupr_sd:.4f}   "
              f"{r.auroc_mean:.4f} +/- {r.auroc_sd:.4f}")

    best = agg.iloc[0]
    print(f"\nbest config      : {best.topology}/{best.fusion} @ top_x={best.top_x:g}"
          f"  AUPR {best.aupr_mean:.4f} +/- {best.aupr_sd:.4f}")
    print(f"no-GCN control   : AUPR {r0['aupr']:.4f}")
    print(f"best single input: {best_in['method']} AUPR {best_in['aupr']:.4f}")
    print(f"mean single input: AUPR {mean_in:.4f}")

    margin = best.aupr_mean - r0["aupr"]
    print(f"\nbest config vs no-GCN control: {margin:+.4f} AUPR "
          f"({margin / best.aupr_sd:.1f} sd)" if best.aupr_sd else "")
    if margin <= 2 * (best.aupr_sd or 0):
        print("  -> not separable from the control at this replication level; "
              "the GCN is not demonstrably contributing.")
    if best.aupr_mean < best_in["aupr"]:
        print(f"  -> still below the best single input by "
              f"{best_in['aupr'] - best.aupr_mean:.4f} AUPR.")


if __name__ == "__main__":
    main()
