"""
run_full_ensemble_net3.py

Standalone script -- run this directly on your machine, no MultiGCN package
dependency needed beyond torch_geometric. Reads every algorithm matrix CSV
from a folder (e.g. dream4_net3_out/matrices/input/), runs the validated
GCN-GAE ensemble (top-10% input threshold, 3-layer GCN, bilinear + distance
decoders averaged, 300 epochs, 3 iterations), and evaluates against your
gold standard.

Usage:
    python run_full_ensemble_net3.py \\
        --input-dir dream4_net3_out/matrices/input \\
        --gold-standard path/to/dream4_100_03_goldstandard.tsv \\
        --gold-standard-format matrix   # or "edgelist"
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_PCT = 10.0
EMBED_DIM = 16
EPOCHS = 300
NUM_ITERATIONS = 3
SEED = 42


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_matrix(path):
    df = pd.read_csv(path, index_col=0)
    return df.values, list(df.index)


def load_gold_standard(path, gene_names, fmt='matrix'):
    num_genes = len(gene_names)
    if fmt == 'matrix':
        gt = pd.read_csv(path, index_col=0)
        gt_matrix = gt.loc[gene_names, gene_names].values.astype(int)
    else:  # edgelist: source,target,label -- no header
        df = pd.read_csv(path, sep=None, engine='python', header=None,
                          names=['source', 'target', 'label'])
        index = {g: i for i, g in enumerate(gene_names)}
        df = df[(df['label'] > 0) & df['source'].isin(index) & df['target'].isin(index)]
        gt_matrix = np.zeros((num_genes, num_genes), dtype=np.int8)
        gt_matrix[df['source'].map(index), df['target'].map(index)] = 1
    return gt_matrix


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-5, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    identity = sp.eye(adj.shape[0])
    laplacian = identity - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return laplacian.tocoo()


def build_edge_index_topX(alg_matrix, top_pct=TOP_PCT):
    n = alg_matrix.shape[0]
    triu = np.triu_indices(n, k=1)
    scores = alg_matrix[triu]
    n_keep = max(1, int(len(scores) * top_pct / 100))
    top_idx = np.argsort(scores)[::-1][:n_keep]
    binary = np.zeros_like(alg_matrix)
    for idx in top_idx:
        i, j = triu[0][idx], triu[1][idx]
        binary[i, j] = 1
        binary[j, i] = 1
    adj_sparse = sp.csr_matrix(binary)
    adj_normalized = normalize_adj(adj_sparse + sp.eye(adj_sparse.shape[0]))
    edge_index, _ = from_scipy_sparse_matrix(adj_normalized)
    return edge_index


class GCNEncoder(torch.nn.Module):
    """3-layer GCN encoder, widths (32,32,16) -- validated across this project."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.01):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * hidden_channels, cached=False)
        self.conv2 = GCNConv(2 * hidden_channels, 2 * hidden_channels, cached=False)
        self.conv3 = GCNConv(2 * hidden_channels, out_channels, cached=False)
        self.bn1 = BatchNorm1d(2 * hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.bn1(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2) + x1
        x3 = F.relu(self.conv3(x2, edge_index))
        return x3


class BilinearDecoder(torch.nn.Module):
    """score = sigmoid(z_i^T W z_j) -- validated best single decoder."""

    def __init__(self, embed_dim):
        super().__init__()
        self.W = torch.nn.Parameter(torch.eye(embed_dim) + 0.01 * torch.randn(embed_dim, embed_dim))

    def forward(self, z, edge_index, sigmoid=True):
        src, dst = edge_index
        value = torch.sum((z[src] @ self.W) * z[dst], dim=-1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        value = (z @ self.W) @ z.t()
        return torch.sigmoid(value) if sigmoid else value


class DistanceDecoder(torch.nn.Module):
    """score = sigmoid(-||z_i - z_j||^2) -- second ensemble member."""

    def forward(self, z, edge_index, sigmoid=True):
        src, dst = edge_index
        dist_sq = ((z[src] - z[dst]) ** 2).sum(dim=-1)
        value = -dist_sq
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        dist_sq = torch.cdist(z, z) ** 2
        value = -dist_sq
        return torch.sigmoid(value) if sigmoid else value


def train_model(decoder, edge_index, x, num_genes, epochs=EPOCHS, weight_decay=1e-8):
    encoder = GCNEncoder(num_genes, EMBED_DIM, EMBED_DIM)
    model = GAE(encoder, decoder).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
        recon = model.decoder.forward_all(z).cpu().numpy()
    return (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)


def evaluate(fused, y_true, triu, label):
    auroc = roc_auc_score(y_true, fused[triu])
    aupr = average_precision_score(y_true, fused[triu])
    print(f"  [{label}] AUROC={auroc:.4f}  AUPR={aupr:.4f}")
    return auroc, aupr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True,
                         help="Folder containing one CSV matrix per algorithm")
    parser.add_argument('--gold-standard', required=True)
    parser.add_argument('--gold-standard-format', choices=['matrix', 'edgelist'], default='matrix')
    parser.add_argument('--top-pct', type=float, default=TOP_PCT)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--iterations', type=int, default=NUM_ITERATIONS)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Device: {DEVICE}")

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    algorithms = {}
    gene_names = None
    for f in files:
        alg_name = os.path.splitext(os.path.basename(f))[0]
        mat, genes = load_matrix(f)
        algorithms[alg_name] = mat
        gene_names = genes
    num_genes = len(gene_names)

    print(f"Loaded {len(algorithms)} algorithms: {list(algorithms.keys())}")

    node_features = np.identity(num_genes)
    x_tensor = torch.tensor(node_features, dtype=torch.float).to(DEVICE)

    gt_matrix = load_gold_standard(args.gold_standard, gene_names, fmt=args.gold_standard_format)
    triu = np.triu_indices(num_genes, k=1)
    y_true = gt_matrix[triu]
    print(f"Gold standard: {y_true.sum()} true edges out of {len(y_true)} possible pairs "
          f"({100*y_true.sum()/len(y_true):.2f}% density)")

    print(f"\n=== Training {len(algorithms)} algorithms x {args.iterations} iterations x 2 decoders "
          f"(bilinear + distance) ===")
    t0 = time.time()
    per_alg_avg_scores = {}
    all_ensemble_recons = []

    for alg_name, mat in algorithms.items():
        edge_index = build_edge_index_topX(mat, top_pct=args.top_pct).to(DEVICE)
        bilinear_recons, distance_recons = [], []
        for _ in range(args.iterations):
            r_bi = train_model(BilinearDecoder(EMBED_DIM), edge_index, x_tensor, num_genes, epochs=args.epochs)
            r_di = train_model(DistanceDecoder(), edge_index, x_tensor, num_genes, epochs=args.epochs)
            bilinear_recons.append(r_bi)
            distance_recons.append(r_di)
            all_ensemble_recons.append(r_bi)
            all_ensemble_recons.append(r_di)
        per_alg_avg_scores[alg_name] = np.mean(bilinear_recons + distance_recons, axis=0)
        print(f"  {alg_name} done ({time.time()-t0:.0f}s elapsed)")

    print(f"\nTotal training time: {time.time()-t0:.0f}s across {len(all_ensemble_recons)} training runs")

    fused_all = np.mean(all_ensemble_recons, axis=0)

    print("\n=== Individual algorithm performance (each alone, bilinear+distance averaged) ===")
    individual_results = []
    for alg_name, scores in sorted(per_alg_avg_scores.items()):
        auroc, aupr = evaluate(scores, y_true, triu, alg_name)
        individual_results.append((alg_name, auroc, aupr))

    print(f"\n=== FULL {len(algorithms)}-ALGORITHM FUSED RESULT ===")
    fused_auroc, fused_aupr = evaluate(fused_all, y_true, triu, f"MultiGCN fused (all {len(algorithms)} algorithms)")

    print("\n=== SUMMARY TABLE (sorted by AUPR) ===")
    print(f"{'Method':<20}{'AUROC':<10}{'AUPR'}")
    for alg_name, auroc, aupr in sorted(individual_results, key=lambda x: -x[2]):
        print(f"{alg_name:<20}{auroc:<10.4f}{aupr:.4f}")
    print(f"{'FUSED (all)':<20}{fused_auroc:<10.4f}{fused_aupr:.4f}")

    # Save the final fused network
    out_path = 'fused_network_full_ensemble.csv'
    pd.DataFrame(fused_all, index=gene_names, columns=gene_names).to_csv(out_path)
    print(f"\nSaved fused network to {out_path}")


if __name__ == '__main__':
    main()
