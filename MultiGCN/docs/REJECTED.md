# Tested and rejected

Everything below was implemented and empirically tested, not merely considered.
Re-testing any of it without new evidence repeats finished work.

## Data / input

| Idea | Result | Why rejected |
|---|---|---|
| `normalized_weight_matrix_*.csv` as score source | Invalid | Pre-binarised despite the name. Threshold sweeps on it gave identical edge counts at 0.2/0.4/0.6/0.8. Always use raw `GRN_*.csv`. |
| Fixed absolute score cutoff (0.6, 0.15) | Not comparable | At 0.6, LEAP kept ~67% of pairs while others kept <0.3%. |
| Top-20%+ input density (with bilinear decoder) | Clearly worse, monotonically | The "5-20% stable" plateau was specific to the old decoder and did not transfer. |
| Native-weight edge confidence | Inconclusive | Initially promising, but that test ran on a pre-binarised file. On real data, a wash. |
| Correlation-weighted edges | No improvement | Tied or slightly worse. |
| Absolute 0.2 cutoff instead of top-x (2026-11, 23 algorithms, all fusion variants) | Worse AUPR in every variant | Kept 0.4% (PLSNET) to 87% (PUC) of pairs depending on the algorithm. Per-separate-GCN AUPR 0.159 -> 0.152, worse on 5/5 nets. |
| Rank-weighted edges into GCNConv (keep_weights, 2026-11 rerun) | Tie (0.157 vs 0.158) | Edge confidence adds nothing the top-x pattern lacks. |

## Architecture

| Idea | Result | Why rejected |
|---|---|---|
| Real expression as node features | Clearly worse on all metrics | Identity forces pure-topology learning, which matches the reconstruction objective. |
| GAT (attention) encoder | Worse, ~67% slower | Too few neighbours per gene at this density for attention to beat uniform averaging. |
| GraphSAGE encoder (alone, and + bilinear) | Clearly worse both times | — |
| Jumping Knowledge | Worse, no gain from more training | Genuine underperformer, not undertrained. |
| 4-layer (32,32,16,8) | Clearly worse | — |
| Alternative widths (funnel, wide-middle, bottleneck, constant-wide) | None beat (32,32,16) | At any epoch count tested. |
| MLP decoder | Beat baseline AUPR only at 450 epochs | Never beat bilinear or the bilinear+distance ensemble. |
| Joint/shared encoder across algorithms | Roughly tied | Needed more epochs just to reach parity with independent training. Superseded: the multi-relational version (one edge type and one DistMult decoder per algorithm, shared bases) is now the default - see below. |
| **One GAE per algorithm** (previous default) | Superseded (2026-11) | Each GAE only sees and rebuilds its own graph, so it can only smooth it: it tied the closed-form Katz index (0.158 vs 0.157 AUPR), and its outputs added nothing to the raw scores in a supervised leave-one-network-out test (0.166 vs 0.165). The joint model beat it on 5/5 DREAM4 nets. |
| APPNP encoder (hop k weighted 0.1 x 0.9^k, 10 hops) | Worse (0.149 vs 0.158) | No better than the rank-average; propagating 10 hops blurs local structure. |
| Joint model without basis sharing (one weight set per algorithm) | Catastrophic (0.107 vs 0.164) | Back to independent graphs. Sharing weights across algorithms is what makes the joint model work; 4 or 8 bases tie. |
| Joint model: longer training / larger embeddings (600-1200 epochs, dim 32-64) | Worse (down to 0.151) | Training loss falls 1.04 -> 0.44 while AUPR falls: it memorises each graph. Tuned on DREAM4 nets 1-3; 100-300 epochs at dim 16 form a plateau, and the best of it tied the default on held-out nets 4-5 (0.182 vs 0.184). 3 layers also worse. |

## Training

| Idea | Result | Why rejected |
|---|---|---|
| Model-score hard-negative mining | Catastrophic (near-random) | Punishes the encoder for confidently proposing genuine hidden links — the exact behaviour the method relies on. |
| Structural (common-neighbour) hard negatives | Helped 1 network, hurt 2 | Does not generalise. |
| Focal Loss (γ=2, α=0.25) | Won 1, lost 1, split 1 | Not a consistent win. |

## Fusion

| Idea | Result | Why rejected |
|---|---|---|
| Binarise-then-vote-count | ~0.02 AUROC worse | Controlled test, identical trained models. Discards confidence. |
| Borda count / rank fusion | Clearly worse | Keeps rank order, discards magnitude. |
| SNF | Worse AUROC, better AUPR | Mixed; not adopted as default. Possibly worth revisiting if AUPR is the sole priority. |
| NMF fusion, entropy-weighted fusion, Jaccard filtering | Tied with simple averaging | — |
| Entropy-weighted + Jaccard combined | Catastrophic (AUROC ~0.61) | The entropy formula rewards single-algorithm-confident edges, not cross-algorithm agreement — contrary to its own inline documentation. |
| Grouping redundant algorithms before fusion | Wash or slightly worse | Flat averaging already handles redundancy; explicit grouping removes reinforcement. Re-tested 2026-11 with 23 algorithms, by registry family and by unsupervised clustering (5-12 groups): worse on 5/5 nets (0.147 vs 0.159). The algorithms that sit alone (LEAP, MEOMI, NONLINEARODES) are the weakest, and grouping hands each a full group's vote. Agreement weighting does the opposite and is now the default. |
| Blending the model with the Katz index | Tie | Katz uses the same neighbourhood structure; blending with the raw-score rank-average is what helps (AUROC). |
| Max instead of mean within groups | Clearly worse everywhere | Far more sensitive to single-algorithm noise spikes. |
| Algorithm ablation | Non-monotonic | Below ~8 algorithms clearly hurts; partial removal down to ~10 is sometimes neutral or slightly better. Real redundancy exists, but "more is always better" is not proven. |

## Open items

1. **E. coli / DREAM5 validation** — highest priority. Every result above is from
   the DREAM4 synthetic family.
2. Whether the 2–3× density multiplier transfers to E. coli's real density.
3. DREAM4 network 2 resisted nearly every improvement that helped on 1 and 3;
   the reason was never identified.
4. PPI-filtered 2-hop neighbour construction (STRING) — untestable on synthetic
   DREAM4 genes; testable on E. coli. Open concern: does PPI filtering recover
   genuine regulation, or conflate it with pathway co-membership?
5. Tukey `k` never validated for the bilinear+distance **ensemble** output
   specifically (only plain k=3.0, bilinear k=1.5).
6. No formal held-out validation split yet. Build one before reporting
   paper-ready numbers.
