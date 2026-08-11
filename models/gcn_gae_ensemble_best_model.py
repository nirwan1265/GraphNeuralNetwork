"""
gcn_gae_ensemble_best_model.py

THE NEW BASE MODEL, replacing gcn_gae_bilinear_model.py as the best-validated
pipeline. Adds one final change on top of everything in that file: instead
of a single decoder, this trains BOTH a bilinear decoder and a distance
decoder per algorithm, and averages their outputs together.

WHY: bilinear and distance decoders are two structurally different ideas
(bilinear learns an asymmetric interaction matrix; distance measures plain
closeness in embedding space) that each independently beat the old
inner-product decoder, but neither was consistently the single best option
across all 3 DREAM4 networks tested. Averaging them together -- the same
logic as averaging 16 different algorithms, just one level up -- turned out
to beat BOTH individual decoders on AUPR on every network tested, and beat
both on AUROC on 2 of 3 networks:

    Network   Bilinear alone     Distance alone      Ensemble (averaged)
    Net 1     0.7861 / 0.1163    0.7609 / 0.1208      0.7825 / 0.1251
    Net 2     0.6794 / 0.0936    0.6730 / 0.0852      0.6824 / 0.0956
    Net 3     0.6986 / 0.1316    0.6991 / 0.1491      0.7076 / 0.1774

This is the most consistent result found across this whole project's
architecture testing -- more consistent than either decoder was alone.

OTHER THINGS TESTED ALONGSIDE THIS AND NOT ADOPTED (all on Network 3):
  - Native-weight edge confidence + bilinear: no improvement (0.1206 vs
    0.1316 baseline AUPR).
  - GraphSAGE encoder + bilinear: clear loss (0.0982 vs 0.1316 AUPR),
    consistent with GraphSAGE losing paired with the old decoder too.
  - A deeper, narrower 4-layer width (32,32,16,8) + bilinear: clear loss
    (0.0979 vs 0.1316 AUPR) -- consistent with the earlier width sweep
    finding that the (32,32,16) 3-layer shape is already a good choice.

Everything else is unchanged from gcn_gae_bilinear_model.py / the base
pipeline validated across this project:
  - Real continuous per-algorithm confidence scores (GENECI's raw "lists/"
    output, not the pre-binarized "normalized_weight_matrix" files)
  - Input topology: top-10% per algorithm (rank-based, validated to be
    roughly 2-3x each network's own true regulatory density)
  - 3-layer GCN encoder, widths (32, 32, 16)
  - Continuous averaging fusion across all (algorithm x iteration) outputs
  - 300 default training epochs

Usage:
    python gcn_gae_ensemble_best_model.py \\
        --lists-dir path/to/inferred_networks/<dataset>/lists \\
        --ground-truth path/to/gold_standard.csv \\
        --output fused_network.csv \\
        --num-genes 100
"""

import argparse
import glob
import os
import time
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import from_scipy_sparse_matrix

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    _HAS_SKLEARN_METRICS = True
except ImportError:
    _HAS_SKLEARN_METRICS = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_TOP_PCT = 10.0
NUM_ITERATIONS = 3
EPOCHS = 300
EMBEDDING_DIM = 16
USE_NATIVE_EDGE_WEIGHTS = False  # tested with bilinear specifically -- no improvement
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_algorithm_matrices(lists_dir, num_genes, gene_prefix='G'):
    """Loads GENECI's raw per-algorithm confidence lists (continuous scores,
    before any thresholding) into full (num_genes, num_genes) matrices."""
    gene_names = [f"{gene_prefix}{i}" for i in range(1, num_genes + 1)]
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    files = sorted(glob.glob(os.path.join(lists_dir, 'GRN_*.csv')))
    if not files:
        raise FileNotFoundError(f"No GRN_*.csv files found in {lists_dir}")

    matrices = {}
    for path in files:
        alg = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, header=None, names=['source', 'target', 'score'])
        mat = np.zeros((num_genes, num_genes))
        for _, row in df.iterrows():
            if row['source'] in gene_idx and row['target'] in gene_idx:
                i, j = gene_idx[row['source']], gene_idx[row['target']]
                mat[i, j] = row['score']
        matrices[alg] = mat
    return matrices, gene_names


def normalize_adj(adj):
    """Normalized Laplacian, used only to determine which edges exist (topology)."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-5, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    identity = sp.eye(adj.shape[0])
    laplacian = identity - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return laplacian.tocoo()


def build_edge_index_topX(alg_matrix, top_pct=INPUT_TOP_PCT, use_native_weights=USE_NATIVE_EDGE_WEIGHTS):
    """Rank-based per-algorithm topology: keep this algorithm's own top X% of
    edges by score, rather than a fixed absolute cutoff."""
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

    if not use_native_weights:
        return edge_index, None
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    confidence = np.clip(alg_matrix[src, dst], 0.05, None)
    edge_weight = torch.tensor(confidence, dtype=torch.float)
    return edge_index, edge_weight


class GCNEncoder(torch.nn.Module):
    """3-layer GCN encoder, widths (2*hidden, 2*hidden, out) -- validated
    against multiple depth/width/encoder-type alternatives."""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.01):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * hidden_channels, cached=False)
        self.conv2 = GCNConv(2 * hidden_channels, 2 * hidden_channels, cached=False)
        self.conv3 = GCNConv(2 * hidden_channels, out_channels, cached=False)
        self.bn1 = BatchNorm1d(2 * hidden_channels, momentum=0.1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, edge_weight=None):
        x1 = F.relu(self.conv1(x, edge_index, edge_weight))
        x2 = self.conv2(x1, edge_index, edge_weight)
        x2 = F.relu(self.bn1(x2))
        x2 = self.dropout(x2) + x1
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        return x3


class BilinearDecoder(torch.nn.Module):
    """score = sigmoid(z_i^T W z_j). W learned, initialized near identity."""

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
    """score = sigmoid(-||z_i - z_j||^2)."""

    def forward(self, z, edge_index, sigmoid=True):
        src, dst = edge_index
        dist_sq = ((z[src] - z[dst]) ** 2).sum(dim=-1)
        value = -dist_sq
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        dist_sq = torch.cdist(z, z) ** 2
        value = -dist_sq
        return torch.sigmoid(value) if sigmoid else value


def train_gae(edge_index, edge_weight, x, device, decoder, embedding_dim=EMBEDDING_DIM,
              epochs=EPOCHS, weight_decay=1e-8):
    """Trains one GCN-GAE with the given decoder instance and returns its
    normalized [0,1] reconstructed adjacency."""
    encoder = GCNEncoder(in_channels=x.shape[1], hidden_channels=embedding_dim, out_channels=embedding_dim)
    model = GAE(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z = model.encode(x, edge_index, edge_weight) if edge_weight is not None else model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index, edge_weight) if edge_weight is not None else model.encode(x, edge_index)
        recon = model.decoder.forward_all(z).cpu().numpy()
    return (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)


def run_pipeline(lists_dir, num_genes=100, top_pct=INPUT_TOP_PCT, num_iterations=NUM_ITERATIONS,
                  epochs=EPOCHS, embedding_dim=EMBEDDING_DIM,
                  use_native_weights=USE_NATIVE_EDGE_WEIGHTS, seed=SEED):
    """Runs the full ensemble pipeline: trains both a bilinear-decoder GAE and
    a distance-decoder GAE per algorithm/iteration, then averages ALL of it
    together (both decoders, all algorithms, all iterations) into one fused
    continuous adjacency matrix."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    matrices, gene_names = load_algorithm_matrices(lists_dir, num_genes)
    algorithms = list(matrices.keys())
    node_features = np.identity(num_genes)
    x_tensor = torch.tensor(node_features, dtype=torch.float).to(device)

    print(f"Genes: {num_genes} | Algorithms: {len(algorithms)} | Top-{top_pct}% input | "
          f"Decoders: bilinear + distance (ensemble) | Iterations/algorithm: {num_iterations} | Epochs: {epochs}")

    t0 = time.time()
    fused = np.zeros((num_genes, num_genes))
    n_runs = 0
    for alg in algorithms:
        mat = matrices[alg]
        edge_index, edge_weight = build_edge_index_topX(mat, top_pct=top_pct, use_native_weights=use_native_weights)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device) if edge_weight is not None else None
        for _ in range(num_iterations):
            recon_bilinear = train_gae(edge_index, edge_weight, x_tensor, device,
                                        BilinearDecoder(embedding_dim),
                                        embedding_dim=embedding_dim, epochs=epochs)
            recon_distance = train_gae(edge_index, edge_weight, x_tensor, device,
                                        DistanceDecoder(),
                                        embedding_dim=embedding_dim, epochs=epochs)
            fused += recon_bilinear
            fused += recon_distance
            n_runs += 2
        print(f"  {alg} done")
    fused /= n_runs
    print(f"Total time: {time.time() - t0:.1f}s across {n_runs} training runs (bilinear + distance, "
          f"{num_iterations} iterations x {len(algorithms)} algorithms x 2 decoders)")

    return fused, gene_names


def evaluate(fused_adjacency, gene_names, ground_truth_path):
    """Computes AUROC/AUPR/best-F1 against a gold-standard network, if provided."""
    if not _HAS_SKLEARN_METRICS:
        print("scikit-learn not available -- skipping evaluation.")
        return None

    ground_truth = pd.read_csv(ground_truth_path, index_col=0)
    num_genes = len(gene_names)
    gt_matrix = ground_truth.loc[gene_names, gene_names].values.astype(int)
    triu = np.triu_indices(num_genes, k=1)
    y_true = gt_matrix[triu]
    y_score = fused_adjacency[triu]

    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    best_f1 = max(
        f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        for t in np.linspace(y_score.min(), y_score.max(), 50)
    )
    print(f"AUROC={auroc:.4f}  AUPR={aupr:.4f}  best-F1={best_f1:.4f}")
    return {'auroc': auroc, 'aupr': aupr, 'best_f1': best_f1}


def main():
    parser = argparse.ArgumentParser(description="Ensemble GCN-GAE model (bilinear + distance decoders averaged)")
    parser.add_argument('--lists-dir', required=True,
                         help="Path to GENECI's lists/ folder containing GRN_<ALGORITHM>.csv edge-list files")
    parser.add_argument('--ground-truth', default=None,
                         help="Optional path to a gold-standard network CSV for AUROC/AUPR/F1 evaluation")
    parser.add_argument('--output', default='fused_network.csv',
                         help="Where to save the fused continuous adjacency matrix")
    parser.add_argument('--num-genes', type=int, default=100)
    parser.add_argument('--top-pct', type=float, default=INPUT_TOP_PCT)
    parser.add_argument('--iterations', type=int, default=NUM_ITERATIONS)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--embedding-dim', type=int, default=EMBEDDING_DIM)
    parser.add_argument('--native-weights', action='store_true',
                         help="Enable native-weight edge confidence (tested with bilinear alone, no improvement)")
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    fused, gene_names = run_pipeline(
        args.lists_dir,
        num_genes=args.num_genes,
        top_pct=args.top_pct,
        num_iterations=args.iterations,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        use_native_weights=args.native_weights,
        seed=args.seed,
    )

    pd.DataFrame(fused, index=gene_names, columns=gene_names).to_csv(args.output)
    print(f"Saved fused network to {args.output}")

    if args.ground_truth:
        evaluate(fused, gene_names, args.ground_truth)


if __name__ == '__main__':
    main()
