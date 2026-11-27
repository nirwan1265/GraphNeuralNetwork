#!/usr/bin/env python
"""
Convert a DREAM5 network directory into the layout MultiGCN expects.

Three format differences from DREAM4, each of which silently corrupts results
if ignored:

  1. Orientation. DREAM5 expression is samples x genes (genes across the
     header). read_expression() expects genes x samples. Detected and
     transposed here rather than assumed.

  2. Gold standard coverage. DREAM5 lists only TF -> target pairs, and the
     official evaluation scores only the TF x gene submatrix. Scoring all
     gene pairs counts pairs the gold standard never assessed as negatives,
     which deflates AUPR. The TF list is copied out so evaluation can be
     restricted.

  3. Scale. 1643 genes vs 100 means ~270x more pairs. Reported below so the
     runtime is not a surprise.

Usage:
    python experiments/prep_dream5.py \
        --network-dir "~/Documents/Github/GraphNeuralNetwork/data/e_coli/DREAM5_network_inference_challenge/Network1" \
        --out-dir ~/Documents/Github/GraphNeuralNetwork/data/dream5_net1_prepped
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network-dir", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    args = p.parse_args()

    root = Path(args.network_dir).expanduser()
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    expr_path = next((root / "input data").glob("*expression_data.tsv"))
    tf_path = next((root / "input data").glob("*transcription_factors.tsv"))
    gs_path = next(g for g in (root / "gold standard").glob("*.tsv")
                   if "signed" not in g.name)

    # --- expression ------------------------------------------------------
    expr = pd.read_csv(expr_path, sep="\t")
    print(f"read {expr_path.name}: {expr.shape[0]} rows x {expr.shape[1]} cols")

    # DREAM5 puts genes in the header. If the columns look like gene ids and
    # there are more columns than rows, it is samples x genes -> transpose.
    looks_like_genes = sum(str(c).startswith("G") for c in expr.columns[:20]) > 10
    if looks_like_genes:
        print("  header looks like gene ids -> transposing to genes x samples")
        expr = expr.T
    else:
        print("  header does not look like gene ids -> leaving orientation alone")
    expr.index.name = "gene"
    expr.columns = [f"S{i+1}" for i in range(expr.shape[1])]

    n_genes, n_samples = expr.shape
    expr_out = out / "expression.csv"
    expr.to_csv(expr_out)
    print(f"  -> {expr_out}  ({n_genes} genes x {n_samples} samples)")

    # --- transcription factors -------------------------------------------
    tfs = [ln.strip() for ln in open(tf_path) if ln.strip()]
    tfs = [t for t in tfs if t in set(expr.index)]
    (out / "tfs.txt").write_text("\n".join(tfs) + "\n")
    print(f"  -> {out/'tfs.txt'}  ({len(tfs)} TFs present in expression)")

    # --- gold standard ----------------------------------------------------
    gs = pd.read_csv(gs_path, sep="\t", header=None,
                     names=["source", "target", "label"])
    pos = gs[gs.label == 1]
    gs_out = out / "gold_standard.csv"
    pos[["source", "target"]].assign(weight=1).to_csv(
        gs_out, index=False, header=False)
    print(f"  -> {gs_out}  ({len(pos)} positive edges, "
          f"{gs.shape[0]} rows total)")

    # --- what this is going to cost --------------------------------------
    pairs = n_genes * (n_genes - 1) // 2
    tf_pairs = len(tfs) * n_genes
    print(f"\nscale check")
    print(f"  unordered gene pairs      {pairs:>12,}")
    print(f"  TF x gene pairs (official) {tf_pairs:>12,}")
    print(f"  positives                 {len(pos):>12,}"
          f"   ({100*len(pos)/tf_pairs:.3f}% of TF x gene)")
    print(f"  dense float32 matrix      {n_genes**2*4/1e6:>12.1f} MB each")
    print(f"\nAt 48 reconstructions that is "
          f"{48*n_genes**2*4/1e9:.1f} GB held in memory by the harness.")
    print("Reduce with --decoders bilinear --seeds 0 if that is too much.")

    print("\nNOTE: no time column is present in DREAM5 expression data, so "
          "SWING, DYNGENIE3, NETWORKBMA, G1DBN and LEAP cannot run. Timing "
          "metadata lives in chip_features.tsv and would need separate "
          "reconstruction.")


if __name__ == "__main__":
    main()
