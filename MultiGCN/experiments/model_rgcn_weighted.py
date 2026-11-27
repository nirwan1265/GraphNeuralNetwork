#!/usr/bin/env python
"""Model 1/3: learned per-algorithm relation weights (R-GCN style).

Leave-one-out showed deleting SWING (best input, AUPR 0.38/0.19/0.30) moved
fused AUPR by <1 sd. Uniform averaging is insensitive to its best input.

This replaces "21 autoencoders, average outputs" with ONE autoencoder over a
multi-relational graph: each algorithm is a relation with one learnable scalar.
Convolutions are SHARED across relations, so the only per-algorithm parameters
are those scalars. Output is one fused network, no post-hoc averaging.

RISK: weights come from reconstruction loss alone (no gold standard), but that
rewards agreement with the CONSENSUS, not accuracy. SWING is good precisely
because it is a minority signal. So this may down-weight it and up-weight the
redundant minet/tree families. The check: correlate learned weights against
true AUPR afterwards -- gold standard validates, never trains.
"""
from __future__ import annotations
import argparse, os, random, sys, time
from pathlib import Path
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variants import build_graph, evaluation_mask, masked_metrics, minmax, raw_consensus
from multigcn.data.loaders import load_edge_list, read_expression
from multigcn.evaluation.metrics import load_gold_standard
from multigcn.models.decoders import build_decoder
from torch_geometric.nn import GAE, GCNConv


class RelationWeightedEncoder(torch.nn.Module):
    def __init__(self, n_rel, in_ch, hidden=16, out_ch=16, dropout=0.01):
        super().__init__()
        self.conv1 = GCNConv(in_ch, 2*hidden, cached=False)
        self.conv2 = GCNConv(2*hidden, 2*hidden, cached=False)
        self.conv3 = GCNConv(2*hidden, out_ch, cached=False)
        self.bn1 = torch.nn.BatchNorm1d(2*hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        # zeros -> uniform softmax: starts at current production behaviour
        self.logits = torch.nn.Parameter(torch.zeros(n_rel))

    def weights(self):
        return torch.softmax(self.logits, dim=0)

    def _mix(self, conv, x, graphs, w):
        out = None
        for r, (ei, ew) in enumerate(graphs):
            h = conv(x, ei, ew)
            out = w[r]*h if out is None else out + w[r]*h
        return out

    def forward(self, x, graphs):
        w = self.weights()
        x1 = F.relu(self._mix(self.conv1, x, graphs, w))
        x2 = F.relu(self.bn1(self._mix(self.conv2, x1, graphs, w)))
        x2 = self.dropout(x2) + x1
        return F.relu(self._mix(self.conv3, x2, graphs, w))


def train(graphs, n, decoder, epochs, lr, wd, hidden, out_ch, dropout, seed, device):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dev = torch.device(device or "cpu")
    graphs = [(ei.to(dev), None if ew is None else ew.to(dev)) for ei, ew in graphs]
    x = torch.eye(n, device=dev)
    enc = RelationWeightedEncoder(len(graphs), n, hidden, out_ch, dropout)
    model = GAE(enc, build_decoder(decoder, out_ch)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        z = model.encoder(x, graphs)
        # every edge set is a target -- this is what makes weights consensus-seeking
        loss = sum(model.recon_loss(z, ei) for ei, _ in graphs) / len(graphs)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        z = model.encoder(x, graphs)
        return minmax(model.decoder.forward_all(z).cpu().numpy()), enc.weights().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lists", required=True, type=Path)
    p.add_argument("--expression", required=True, type=Path)
    p.add_argument("--gold", required=True, type=Path)
    p.add_argument("--tfs", type=Path, default=None)
    p.add_argument("--top-x", type=float, default=0.08)
    p.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    p.add_argument("--decoders", nargs="+", default=["bilinear","distance"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=Path("experiments/rgcn_weighted.csv"))
    a = p.parse_args()

    genes = list(read_expression(a.expression).index)
    truth = load_gold_standard(a.gold, genes); n = len(genes)
    if a.tfs:
        pos = {g: i for i, g in enumerate(genes)}
        tf = [pos[g] for g in (l.strip() for l in open(a.tfs)) if g in pos]
        mask = evaluation_mask(n, tf_idx=tf)
    else:
        mask = evaluation_mask(n, undirected=True)
    score = lambda m: masked_metrics(m, truth, mask)

    matrices = {}
    for path in sorted(a.lists.glob("GRN_*.csv")):
        try: m = load_edge_list(path, genes)
        except Exception: continue
        if len(np.unique(m[m > 0])) < 2: continue
        matrices[path.stem[4:]] = m
    names = sorted(matrices)
    print(f"{len(names)} algorithms: {', '.join(names)}")

    per = {k: score(matrices[k])["aupr"] for k in names}
    best = max(per, key=per.get)
    print(f"best single input: {best} AUPR {per[best]:.4f} | mean {np.mean(list(per.values())):.4f}")
    r0 = score(raw_consensus([matrices[k] for k in names]))
    print(f"CONTROL raw rank-average, no GCN: AUPR {r0['aupr']:.4f}\n")
    
    graphs = [build_graph(matrices[k], a.top_x, weighted=True) for k in names]
    rows, all_w = [], []
    for seed in a.seeds:
        t0 = time.time()
        # Pool the decoders into ONE fused network per seed, matching what
        # variants.py does. Scoring each decoder alone is not the pipeline's
        # behaviour and understates bilinear while penalising distance.
        recons, ws = [], []
        for dec in a.decoders:
            fused, w = train(graphs, n, dec, a.epochs, a.lr, a.weight_decay,
                             a.hidden, a.embedding_dim, a.dropout, seed, a.device)
            recons.append(fused); ws.append(w)
        fused = np.mean(np.stack(recons), axis=0)
        r = score(fused)
        rows.append({"seed": seed, "auroc": r["auroc"], "aupr": r["aupr"]})
        all_w.append(np.mean(np.stack(ws), axis=0))
        print(f"  seed={seed}  AUROC {r['auroc']:.4f}  AUPR {r['aupr']:.4f}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows); a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    W = np.stack(all_w); wm, ws = W.mean(0), W.std(0)
    aupr = np.array([per[k] for k in names])

    print(f"\nfused: AUPR {df.aupr.mean():.4f} +/- {df.aupr.std():.4f}   AUROC {df.auroc.mean():.4f} +/- {df.auroc.std():.4f}")
    print(f"control {r0['aupr']:.4f}   best input {best} {per[best]:.4f}")
    print(f"\nlearned weights (uniform = {1/len(names):.4f}):\n")
    order = np.argsort(-wm)
    print(f"{'algorithm':<14}{'weight':>9}{'+/-':>8}{'own AUPR':>11}")
    for i in order:
        print(f"{names[i]:<14}{wm[i]:>9.4f}{ws[i]:>8.4f}{aupr[i]:>11.4f}")

    pr, pp = pearsonr(wm, aupr); sr, sp = spearmanr(wm, aupr)
    print(f"\nweight vs AUPR: pearson {pr:+.3f} (p={pp:.3f})  spearman {sr:+.3f} (p={sp:.3f})")
    if pr > 0.3:   print("  -> weights track accuracy; consensus is a usable proxy.")
    elif pr < -0.3: print("  -> weights ANTI-correlate with accuracy: consensus weighting penalises informative minorities. A finding, not a bug.")
    else:          print("  -> weights carry little accuracy information; no better than uniform.")
    bi = names.index(best)
    print(f"\n{best} got weight {wm[bi]:.4f}, rank {list(order).index(bi)+1} of {len(names)}")
    
    counts = np.array([int((build_graph(matrices[k], a.top_x, weighted=True)[0]
                        .shape[1])) for k in names])
    dpr, dpp = pearsonr(wm, counts)
    print(f"\nweight vs edge count: pearson {dpr:+.3f} (p={dpp:.3f})")
    print(f"edge count vs AUPR:   pearson {pearsonr(counts, aupr)[0]:+.3f}")


if __name__ == "__main__":
    main()
    