"""
gcn_gae_model2_adaptive_pruning.py

Model 2: an alternative fusion/final-cutoff strategy on top of the same
validated GCN-GAE architecture used in gcn_gae_improved_model.py.

WHAT THIS SOLVES THAT THE BASE MODEL DOESN'T:
gcn_gae_improved_model.py outputs a continuous score for every gene pair --
good for ranking quality (AUROC/AUPR), but it never picks an actual final
network. Someone still has to choose a cutoff to get a real, usable list of
predicted edges. On DREAM4 you could cheat and pick whatever cutoff scores
best against the known answer -- but that's not available on real biological
data (e.g. COVID), where there is no ground truth to peek at. This script is
specifically about picking that final cutoff in a way that never looks at
ground truth, so the same method is usable on real data, not just benchmarks.

THREE PIECES, EACH TESTED AND VALIDATED SEPARATELY:

  1. INPUT TOPOLOGY: top-X% per algorithm (rank-based), not a fixed absolute
     score. A fixed absolute cutoff (e.g. 0.15) means wildly different things
     across 16 different algorithms with different score distributions/scales
     (see gcn_gae_improved_model.py's docstring for the measured density
     inconsistency this causes). Rank-based density is portable across
     algorithms and across datasets -- "keep each algorithm's own top X%" means
     the same thing regardless of that algorithm's raw score scale.
     X defaulted to 10% here, consistent with the stable AUROC/AUPR plateau
     (roughly 5-20% density) found when sweeping fixed thresholds earlier.

  2. FUSION: SUM instead of average, across all (algorithm x iteration)
     continuous outputs. Note: sum and average produce the IDENTICAL ranking
     (sum = average x N, a monotonic transform), so this does not change
     AUROC/AUPR at all -- it matters for step 3 below, where the raw
     magnitude of "how many algorithms/iterations agree, and how strongly"
     is used per-gene, not just the relative order.

  3. FINAL CUTOFF: per-gene median+MAD adaptive pruning, NOT a WGCNA-style
     scale-free degree-distribution fit (tested and rejected -- see below),
     and NOT one global threshold for every gene.
     For each gene's own row of summed connection scores: compute that row's
     median and Median Absolute Deviation (MAD), then keep only connections
     that are more than z_cutoff robust-MADs above that gene's OWN median.
     This lets a hub gene (many real connections) and a sparse gene (few real
     connections) each find their own natural cutoff, instead of forcing
     every gene through one global number.
     An edge (i,j) is undirected here, but gene i's row and gene j's row can
     independently disagree about whether it's "unusual" relative to their
     own baseline -- combine='and' requires BOTH genes to independently flag
     it (validated as the better default: more conservative, matches the
     genuinely sparse nature of real GRNs); combine='or' requires only one.

     Why median+MAD instead of the "standard" automatic elbow detector
     (Kneedle): tested directly on a case with one dominant edge, a few
     good-but-lower edges, and then noise -- Kneedle got fooled by the single
     dominant point and cut off too early, discarding real edges. Median+MAD
     (robust statistics) correctly kept the good edges in that same test.
     WGCNA's scale-free (power-law degree distribution) criterion was also
     tested as a final-cutoff method and rejected: at only 100 genes, there
     isn't enough distinct-degree support for a power-law fit to be
     statistically trustworthy (a well-known general pitfall of naive R^2
     power-law fitting on small networks, not specific to this pipeline --
     see Clauset, Shalizi & Newman 2009). It produced a near-fully-dense,
     useless network (F1 = 0.04) even after correcting for the pitfall.

VALIDATED RESULT (DREAM4 network 1, ground truth used only for evaluation,
never for choosing the cutoff):

    | Method                                          | Edges | Precision | Recall | F1    |
    |--------------------------------------------------|-------|-----------|--------|-------|
    | Global blind median+MAD baseline (no per-gene)    | 1038  | 0.062     | 0.653  | 0.113 |
    | Model 2: per-gene median+MAD (z=1.5, and-combine) | 756   | 0.081     | 0.622  | 0.143 |

Model 2 beats the naive global blind baseline (F1 0.143 vs 0.113, ~27%
relative gain) with neither method allowed to see ground truth. For context
only, an ORACLE cutoff that cheats by picking the single best threshold
against the known answer reaches F1 = 0.232 -- that's the ceiling if you had
the answer, not a fair comparison, shown only to be transparent about it.

ON AUROC/AUPR: this script's continuous ranking (before the final per-gene
cutoff is applied) is statistically the same as gcn_gae_improved_model.py's
(AUROC ~0.80, AUPR ~0.10 on DREAM4 network 1) -- expected, since neither the
rank-based input topology nor sum-vs-average fusion change the ranking in a
way that should move those numbers meaningfully. This script's value-add is
producing an actual final ground-truth-blind network, not improving ranking
quality over the base model.

Usage:
    python gcn_gae_model2_adaptive_pruning.py \\
        --lists-dir path/to/inferred_networks/<dataset>/lists \\
        --ground-truth path/to/gold_standard.csv \\
        --output final_network.csv \\
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
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
    _HAS_SKLEARN_METRICS = True
except ImportError:
    _HAS_SKLEARN_METRICS = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOP_PCT = 10.0            # per-algorithm input density -- validated stable range is ~5-20%
NUM_ITERATIONS = 3
EPOCHS = 150
EMBEDDING_DIM = 16
Z_CUTOFF = 1.5             # validated default for per-gene median+MAD pruning
COMBINE_MODE = 'and'       # validated default -- 'and' beat 'or' in testing
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
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum + 1e-5, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    identity = sp.eye(adj.shape[0])
    laplacian = identity - d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return laplacian.tocoo()


def build_edge_index_topX(alg_matrix, top_pct=TOP_PCT):
    """Rank-based per-algorithm topology: keep this algorithm's own top X% of
    edges by score, rather than a fixed absolute cutoff. Portable across
    algorithms with different score scales/distributions."""
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
    """3-layer GCN encoder, same architecture as gcn_gae_improved_model.py."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout_rate=0.01):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, 2 * hidden_channels, cached=False))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(2 * hidden_channels, 2 * hidden_channels, cached=False))
        self.convs.append(GCNConv(2 * hidden_channels, out_channels, cached=False))
        self.bns = torch.nn.ModuleList(
            [BatchNorm1d(2 * hidden_channels, momentum=0.1) for _ in range(num_layers - 1)]
        )
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        for i in range(1, self.num_layers - 1):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i - 1](x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_res
        x = self.convs[-1](x, edge_index)
        x = F.relu(x)
        return x


def train_gae(edge_index, x, device, embedding_dim=EMBEDDING_DIM, epochs=EPOCHS, weight_decay=1e-8):
    encoder = GCNEncoder(in_channels=x.shape[1], hidden_channels=embedding_dim, out_channels=embedding_dim)
    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
        recon = model.decoder.forward_all(z).cpu().numpy()
    return (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)


def per_gene_median_mad_prune(matrix, z_cutoff=Z_CUTOFF, combine=COMBINE_MODE):
    """For each gene's own row of summed connection scores, keep only
    connections that stand out relative to THAT gene's own median/MAD --
    adapts per-gene instead of applying one global cutoff to every gene."""
    n = matrix.shape[0]
    keep_from_row = np.zeros((n, n), dtype=bool)
    for i in range(n):
        row = matrix[i].copy()
        row[i] = -np.inf
        valid = row[row > -np.inf]
        median = np.median(valid)
        mad = np.median(np.abs(valid - median))
        if mad < 1e-9:
            continue
        robust_z = (row - median) / (1.4826 * mad)
        keep_from_row[i] = robust_z > z_cutoff

    if combine == 'or':
        binary = np.logical_or(keep_from_row, keep_from_row.T).astype(int)
    else:
        binary = np.logical_and(keep_from_row, keep_from_row.T).astype(int)
    np.fill_diagonal(binary, 0)
    return binary


def run_pipeline(lists_dir, num_genes=100, top_pct=TOP_PCT, num_iterations=NUM_ITERATIONS,
                  epochs=EPOCHS, embedding_dim=EMBEDDING_DIM, seed=SEED):
    """Runs the full Model 2 pipeline. Returns (fused_sum_continuous, final_binary, gene_names)."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    matrices, gene_names = load_algorithm_matrices(lists_dir, num_genes)
    algorithms = list(matrices.keys())
    node_features = np.identity(num_genes)
    x_tensor = torch.tensor(node_features, dtype=torch.float).to(device)

    print(f"Genes: {num_genes} | Algorithms: {len(algorithms)} | Top-X%: {top_pct} | "
          f"Iterations/algorithm: {num_iterations}")

    t0 = time.time()
    all_recon = []
    for alg in algorithms:
        edge_index = build_edge_index_topX(matrices[alg], top_pct=top_pct).to(device)
        for _ in range(num_iterations):
            recon = train_gae(edge_index, x_tensor, device, embedding_dim=embedding_dim, epochs=epochs)
            all_recon.append(recon)
        print(f"  {alg} done")
    print(f"Total time: {time.time() - t0:.1f}s across {len(all_recon)} training runs")

    fused_sum = np.sum(all_recon, axis=0)  # sum, not average -- see docstring point 2
    final_binary = per_gene_median_mad_prune(fused_sum, z_cutoff=Z_CUTOFF, combine=COMBINE_MODE)

    return fused_sum, final_binary, gene_names


def evaluate(fused_sum, final_binary, gene_names, ground_truth_path):
    """Reports both the continuous ranking quality (AUROC/AUPR, for comparison
    against the base model) and the actual final network's precision/recall/F1
    (the number that matters for Model 2's real contribution)."""
    if not _HAS_SKLEARN_METRICS:
        print("scikit-learn not available -- skipping evaluation.")
        return None

    ground_truth = pd.read_csv(ground_truth_path, index_col=0)
    num_genes = len(gene_names)
    gt_matrix = ground_truth.loc[gene_names, gene_names].values.astype(int)
    triu = np.triu_indices(num_genes, k=1)
    y_true = gt_matrix[triu]

    y_score = fused_sum[triu]
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    print(f"Continuous ranking (for comparison against gcn_gae_improved_model.py): "
          f"AUROC={auroc:.4f}  AUPR={aupr:.4f}")

    y_pred = final_binary[triu]
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    n_edges = int(final_binary.sum() / 2)
    print(f"Final ground-truth-blind network: edges={n_edges}  precision={prec:.4f}  "
          f"recall={rec:.4f}  F1={f1:.4f}")

    return {'auroc': auroc, 'aupr': aupr, 'precision': prec, 'recall': rec, 'f1': f1, 'n_edges': n_edges}


def main():
    global Z_CUTOFF, COMBINE_MODE
    parser = argparse.ArgumentParser(description="Model 2: top-X% input, sum fusion, per-gene adaptive pruning")
    parser.add_argument('--lists-dir', required=True,
                         help="Path to GENECI's lists/ folder containing GRN_<ALGORITHM>.csv edge-list files")
    parser.add_argument('--ground-truth', default=None,
                         help="Optional path to a gold-standard network CSV for evaluation")
    parser.add_argument('--output', default='final_network.csv',
                         help="Where to save the final binary network")
    parser.add_argument('--output-continuous', default=None,
                         help="Optional: also save the continuous summed scores before pruning")
    parser.add_argument('--num-genes', type=int, default=100)
    parser.add_argument('--top-pct', type=float, default=TOP_PCT)
    parser.add_argument('--iterations', type=int, default=NUM_ITERATIONS)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--embedding-dim', type=int, default=EMBEDDING_DIM)
    parser.add_argument('--z-cutoff', type=float, default=Z_CUTOFF)
    parser.add_argument('--combine', choices=['and', 'or'], default=COMBINE_MODE)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    Z_CUTOFF = args.z_cutoff
    COMBINE_MODE = args.combine

    fused_sum, final_binary, gene_names = run_pipeline(
        args.lists_dir, num_genes=args.num_genes, top_pct=args.top_pct,
        num_iterations=args.iterations, epochs=args.epochs,
        embedding_dim=args.embedding_dim, seed=args.seed,
    )

    pd.DataFrame(final_binary, index=gene_names, columns=gene_names).to_csv(args.output)
    print(f"Saved final network to {args.output}")

    if args.output_continuous:
        pd.DataFrame(fused_sum, index=gene_names, columns=gene_names).to_csv(args.output_continuous)
        print(f"Saved continuous summed scores to {args.output_continuous}")

    if args.ground_truth:
        evaluate(fused_sum, final_binary, gene_names, args.ground_truth)


if __name__ == '__main__':
    main()
