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
| Joint/shared encoder across algorithms | Roughly tied | Needed more epochs just to reach parity with independent training. |

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
| Grouping redundant algorithms before fusion | Wash or slightly worse | Flat averaging already handles redundancy; explicit grouping removes reinforcement. |
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
