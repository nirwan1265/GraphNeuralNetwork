"""
gcn_gae_improved_model.py

Improved GCN-GAE gene regulatory network reconstruction model.

This is the second version of this script. The first version read from
the "normalized_weight_matrix_all_GRN_*.csv" files, which turned out to
already be pre-binarized (every value was exactly 0 or 1) despite the
misleading filename. This version reads GENECI's raw per-algorithm
output directly -- the "lists/GRN_<ALGORITHM>.csv" edge-list files
GENECI writes during `geneci infer-network`, before any thresholding --
which are genuinely continuous confidence scores. See
Continuous_Averaging_vs_Binarize_Count.ipynb, SNF_Fusion.ipynb, and
NMF_Entropy_Jaccard_Fusion.ipynb in this folder for the validated
fusion-method comparisons this script's design is based on.

THREE CHANGES FROM THE ORIGINAL FINAL_MODEL_GCN.ipynb, each tested and
validated (or, for #3, corrected after re-testing on real data):

  1. CONTINUOUS-AVERAGING FUSION instead of binarize-then-count.
     The original binarized every algorithm's every training iteration's
     reconstructed adjacency, summed those into a vote count, then
     thresholded the count. Here, reconstructions stay continuous the
     whole way through and are simply averaged.
     Validated: AUROC 0.8056 vs 0.7866 (real ~0.02 gain), tested on
     identical trained models so the difference is attributable to the
     fusion method itself.

  2. INPUT THRESHOLD = 0.15, chosen by an actual sweep, not convention.
     The original notebook used 0.6 with no stated justification. A
     "weak/moderate/strong correlation" convention (e.g. Cohen's
     thresholds) does NOT apply here -- these are 16 different
     algorithms' internal scores (mutual information, tree-ensemble
     importances, partial correlation, etc.), not correlation
     coefficients, and a fixed absolute cutoff means wildly different
     things to each one (e.g. at 0.6, GRN_LEAP alone keeps ~9% of all
     possible edges while several others keep under 0.2%).
     We swept threshold in {0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8} on
     DREAM4 network 1 with ground truth, evaluating AUROC/AUPR/F1 at
     each. Result: performance is a broad, stable plateau across
     roughly 0.05-0.2 (AUROC ~0.796-0.803), then degrades sharply beyond
     0.4, and collapses by 0.8 (AUROC 0.60). The original default of
     0.6 sits past the cliff -- one of the worse points on the whole
     curve, not a safe middle value. 0.15 sits at the peak of the
     stable plateau (AUROC 0.8026) without being at either edge of that
     range. If you change datasets, re-run the sweep in
     sweep_input_threshold() below rather than assuming 0.15 transfers.

  3. NATIVE-WEIGHT EDGE CONFIDENCE -- available but OFF by default.
     Earlier testing (on the pre-binarized file from point 2's mistake)
     showed this consistently improving AUROC/F1. That result doesn't
     hold on genuinely continuous data: re-tested at threshold=0.15 on
     the real confidence-list data, it came out roughly a wash (AUROC
     -0.002, AUPR +0.0004, F1 +0.002 -- all within noise). Left in as an
     option (USE_NATIVE_EDGE_WEIGHTS=True) in case it helps on other
     datasets, but it is not a validated win the way changes #1 and #2
     are, so it defaults to off here.

Usage:
    python gcn_gae_improved_model.py \\
        --lists-dir path/to/inferred_networks/<dataset>/lists \\
        --ground-truth path/to/gold_standard.csv \\
        --output fused_network.csv \\
        --num-genes 100

The --lists-dir should point at GENECI's output folder containing one
GRN_<ALGORITHM>.csv per algorithm, each a 3-column edge list
(source,target,score) with continuous scores in [0,1]. If
--ground-truth is omitted, the script still runs and saves the fused
network without AUROC/AUPR/F1 evaluation (e.g. for a real dataset like
COVID where no gold standard exists).
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
# Configuration -- edit these or override via command-line args
# ---------------------------------------------------------------------------
INPUT_THRESHOLD = 0.15          # validated by sweep -- see docstring point 2
NUM_ITERATIONS = 3               # training runs per algorithm to average over (original notebook used 10;
                                  # 3 was used throughout validation for speed -- raise for a final run)
EPOCHS = 150
EMBEDDING_DIM = 16
USE_NATIVE_EDGE_WEIGHTS = False  # inconclusive on real data -- see docstring point 3
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_algorithm_matrices(lists_dir, num_genes, gene_prefix='G'):
    """Loads GENECI's raw per-algorithm confidence lists into full (num_genes, num_genes)
    continuous matrices. Expects one GRN_<ALGORITHM>.csv per algorithm in lists_dir,
    each a 3-column edge list (source,target,score), no header.
    Missing pairs (not listed) get score 0 -- GENECI only lists edges each algorithm
    considered non-negligible, so absence is itself informative, not missing data.
    """
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


def build_edge_index(alg_matrix, threshold=INPUT_THRESHOLD, use_native_weights=USE_NATIVE_EDGE_WEIGHTS):
    """Builds the graph topology for one algorithm's continuous confidence matrix.

    Topology (which edges exist) is decided by the threshold. If
    use_native_weights, each surviving edge also carries the algorithm's
    own raw continuous score as its edge_weight; otherwise edge_weight is
    None and GCNConv treats every edge as weight=1.
    """
    binary = (alg_matrix >= threshold).astype(float)
    adj_sparse = sp.csr_matrix(binary)
    adj_normalized = normalize_adj(adj_sparse + sp.eye(adj_sparse.shape[0]))
    edge_index, _ = from_scipy_sparse_matrix(adj_normalized)  # Laplacian weight discarded, topology only

    if not use_native_weights:
        return edge_index, None

    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    confidence = alg_matrix[src, dst]
    confidence = np.clip(confidence, 0.05, None)  # floor so no edge goes to zero weight
    edge_weight = torch.tensor(confidence, dtype=torch.float)
    return edge_index, edge_weight


class GCNEncoder(torch.nn.Module):
    """3-layer GCN encoder (same architecture as FINAL_MODEL_GCN.ipynb),
    extended to optionally accept edge_weight."""

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

    def forward(self, x, edge_index, edge_weight=None):
        x = self.convs[0](x, edge_index, edge_weight)
        x = F.relu(x)
        for i in range(1, self.num_layers - 1):
            x_res = x
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.bns[i - 1](x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_res
        x = self.convs[-1](x, edge_index, edge_weight)
        x = F.relu(x)
        return x


def train_gae(edge_index, edge_weight, x, device, embedding_dim=EMBEDDING_DIM, epochs=EPOCHS, weight_decay=1e-8):
    """Trains one GCN-GAE and returns its normalized [0,1] reconstructed adjacency."""
    encoder = GCNEncoder(in_channels=x.shape[1], hidden_channels=embedding_dim, out_channels=embedding_dim)
    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z = model.encode(x, edge_index, edge_weight)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index, edge_weight)
        recon = model.decoder.forward_all(z).cpu().numpy()
    return (recon - recon.min()) / (recon.max() - recon.min() + 1e-12)


def run_pipeline(lists_dir, num_genes=100, num_iterations=NUM_ITERATIONS, epochs=EPOCHS,
                  embedding_dim=EMBEDDING_DIM, use_native_weights=USE_NATIVE_EDGE_WEIGHTS,
                  input_threshold=INPUT_THRESHOLD, seed=SEED):
    """Runs the full improved pipeline and returns (fused_adjacency, gene_names)."""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    matrices, gene_names = load_algorithm_matrices(lists_dir, num_genes)
    algorithms = list(matrices.keys())
    node_features = np.identity(num_genes)  # validated as better than real expression -- see project notes
    x_tensor = torch.tensor(node_features, dtype=torch.float).to(device)

    print(f"Genes: {num_genes} | Algorithms: {len(algorithms)} | Threshold: {input_threshold} | "
          f"Native edge weights: {use_native_weights} | Iterations/algorithm: {num_iterations}")

    t0 = time.time()
    fused = np.zeros((num_genes, num_genes))
    n_runs = 0
    for alg in algorithms:
        mat = matrices[alg]
        edge_index, edge_weight = build_edge_index(mat, threshold=input_threshold,
                                                     use_native_weights=use_native_weights)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device) if edge_weight is not None else None
        for _ in range(num_iterations):
            recon = train_gae(edge_index, edge_weight, x_tensor, device,
                               embedding_dim=embedding_dim, epochs=epochs)
            fused += recon  # continuous averaging -- no binarization before combining
            n_runs += 1
        print(f"  {alg} done")
    fused /= n_runs
    print(f"Total time: {time.time() - t0:.1f}s across {n_runs} training runs")

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


def sweep_input_threshold(lists_dir, ground_truth_path, num_genes=100,
                           thresholds=(0.05, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8),
                           num_iterations=NUM_ITERATIONS, epochs=EPOCHS,
                           embedding_dim=EMBEDDING_DIM, seed=SEED):
    """Re-runs the threshold sensitivity sweep documented in the module docstring.
    Use this if you're on a new dataset and want to check whether 0.15 still sits
    in the stable plateau, rather than assuming it transfers unchanged."""
    results = {}
    for t in thresholds:
        fused, gene_names = run_pipeline(lists_dir, num_genes=num_genes, num_iterations=num_iterations,
                                          epochs=epochs, embedding_dim=embedding_dim,
                                          use_native_weights=False, input_threshold=t, seed=seed)
        print(f"--- threshold={t} ---")
        results[t] = evaluate(fused, gene_names, ground_truth_path)
    print("\n=== SWEEP SUMMARY ===")
    print(f"{'Threshold':<12}{'AUROC':<10}{'AUPR':<10}{'Best F1'}")
    for t, r in results.items():
        print(f"{t:<12}{r['auroc']:<10.4f}{r['aupr']:<10.4f}{r['best_f1']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Improved GCN-GAE gene regulatory network reconstruction")
    parser.add_argument('--lists-dir', required=True,
                         help="Path to GENECI's lists/ folder containing GRN_<ALGORITHM>.csv edge-list files")
    parser.add_argument('--ground-truth', default=None,
                         help="Optional path to a gold-standard network CSV for AUROC/AUPR/F1 evaluation")
    parser.add_argument('--output', default='fused_network.csv',
                         help="Where to save the fused continuous adjacency matrix")
    parser.add_argument('--num-genes', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=NUM_ITERATIONS)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--embedding-dim', type=int, default=EMBEDDING_DIM)
    parser.add_argument('--input-threshold', type=float, default=INPUT_THRESHOLD)
    parser.add_argument('--native-weights', action='store_true',
                         help="Enable native-weight edge confidence (inconclusive on real data so far -- "
                              "see module docstring point 3)")
    parser.add_argument('--sweep-threshold', action='store_true',
                         help="Instead of a single run, sweep thresholds (0.05-0.8) and report AUROC/AUPR/F1 "
                              "for each -- requires --ground-truth")
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    if args.sweep_threshold:
        if not args.ground_truth:
            parser.error("--sweep-threshold requires --ground-truth")
        sweep_input_threshold(args.lists_dir, args.ground_truth, num_genes=args.num_genes,
                               num_iterations=args.iterations, epochs=args.epochs,
                               embedding_dim=args.embedding_dim, seed=args.seed)
        return

    fused, gene_names = run_pipeline(
        args.lists_dir,
        num_genes=args.num_genes,
        num_iterations=args.iterations,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        use_native_weights=args.native_weights,
        input_threshold=args.input_threshold,
        seed=args.seed,
    )

    pd.DataFrame(fused, index=gene_names, columns=gene_names).to_csv(args.output)
    print(f"Saved fused network to {args.output}")

    if args.ground_truth:
        evaluate(fused, gene_names, args.ground_truth)


if __name__ == '__main__':
    main()
