# MultiGCN — project handoff

**Read this first if you are a new assistant session picking up this work.**

> ⚠️ **This file contains unpublished methodology.** The combination approach is
> novel and not yet published. Keep it out of public-facing material (README,
> repo description, conference abstracts) until the paper is out. `README.md` is
> deliberately vaguer than this file — that is intentional, not an oversight.

---

## 1. What this project is

Two parallel goals, sharing one codebase:

1. **A publishable method.** A GCN graph-autoencoder ensemble for gene
   regulatory network (GRN) reconstruction that outperforms existing ensemble
   methods on benchmark data.
2. **A pip-installable package** (`MultiGCN`) that packages the method so
   others can use it, with no dependency on GENECI or any other external tool.

Owner: Nirwan Tandukar, PhD student in computational biology. Also applying to
industry roles (e.g. NVIDIA), so repo quality and README presentation matter.

### The core idea

Every GRN inference algorithm captures a different partial view of the
regulatory structure, with its own blind spots. Instead of picking one, or
naively averaging them:

1. Run **N different inference algorithms** on the same expression data. Each
   produces a continuous confidence score for every gene pair.
2. For each algorithm independently, keep its **top 10% of gene pairs** as a
   graph topology (rank-based, not an absolute cutoff).
3. Train a **separate GCN graph autoencoder** on each of those topologies.
   Node features are the identity matrix — no expression values.
4. Decode with **two decoders** (bilinear + distance) and run **3 iterations**
   each.
5. **Average all reconstructions continuously** into one final network.

The output recovers edges that no individual algorithm proposed.

---

## 2. Current state

### Working and verified

- Package installs (`pip install -e ".[dev]"`), 38 tests pass.
- **36 algorithms registered** across 10 methodological families.
- Full pipeline runs end to end on DREAM4 100-gene network 3: inference →
  topology → GCN training → fusion → scoring.
- GENIE3/GRNBOOST2 reimplemented natively on scikit-learn. Verified against the
  real gold standard: `GENIE3_RF` ntrees=1000 → AUROC 0.742 / AUPR 0.125
  (random baseline AUPR 0.0196).
- The 9 GENECI-derived R wrappers were rewritten as MultiGCN-owned code.
  **`--tier r` no longer needs GENECI at all.**

### Not yet verified

- **No R wrapper has actually been executed successfully end to end** at the
  time of writing (BNLEARN was mid-run). Defaults reproduce GENECI's original
  calls, but the parameter pass-through is untested.
- Docker algorithms: images never pulled or run.
- SWING, dynGENIE3, dynUGENE, networkBMA, ScanBMA, G1DBN, TDARACNE wrappers are
  structure-checked only.
- Some R package argument names (`c3net` alpha/cutoffMI, `kboost` ite,
  `TDARACNE`, `ScanBMAcontrol` gCtrl) were written from memory. The `AddIfSet`
  design means an unset parameter is never passed, so these can only break for
  someone who explicitly overrides them.

### Known outstanding issue

The parent repo (`GraphNeuralNetwork`) had >100 MB generated files
(`tmp-*/GRN_*.csv`) committed, which GitHub rejects. A `.gitignore` and a
`scripts/pre-commit` size guard now exist. History may still need
`git filter-repo` cleanup.

---

## 3. Repository layout

```
GraphNeuralNetwork/                 <- outer repo (research, papers, data)
├── data/Database.DREAM4/
│   ├── EXP/dream4_100_0N_exp.csv   <- expression (genes x samples)
│   └── GS/dream4_100_0N_gs.csv     <- gold standard (genes x genes matrix)
├── scripts/pre-commit              <- blocks >50 MB commits
└── MultiGCN/                       <- the package (NOT a separate git repo)
    ├── multigcn/
    │   ├── registry.py             <- SINGLE SOURCE OF TRUTH for all 36 algorithms
    │   ├── params.py               <- <package>_<param> namespacing + validation
    │   ├── pipeline.py             <- the MultiGCN class
    │   ├── config.py               <- Config dataclass, all validated defaults
    │   ├── cli.py
    │   ├── inference/
    │   │   ├── base.py             <- runner ABC, data-requirement gate
    │   │   ├── runners.py          <- R / Python / Docker runners
    │   │   ├── vendor.py           <- pulls the 4 remaining GENECI scripts
    │   │   └── scripts/
    │   │       ├── r/              <- 17 R wrappers (13 ours, 4 vendored)
    │   │       └── python/         <- GENIE3, DYNGENIE3, SWING
    │   ├── data/
    │   │   ├── loaders.py
    │   │   └── dream4.py           <- DREAM4 time-series conversion
    │   ├── topology/threshold.py   <- rank-based top-X%
    │   ├── models/                 <- encoders.py, decoders.py
    │   ├── training/train.py
    │   ├── fusion/combine.py
    │   ├── postprocessing/tukey.py
    │   └── evaluation/metrics.py
    ├── docs/REJECTED.md            <- READ THIS BEFORE PROPOSING CHANGES
    ├── docs/HANDOFF.md             <- this file
    ├── examples/run_dream4_net3.py
    └── tests/                      <- 38 tests
```

---

## 4. Validated design decisions — do not change casually

Every row below was tested experimentally. Changing one means re-running the
full DREAM4 benchmark.

| Component | Setting | Why |
|---|---|---|
| Input topology | top 10%, rank-based | Fixed cutoffs gave 0.3%–67% density across algorithms at the same threshold value |
| Node features | identity matrix | Real expression values as features tested, clearly worse |
| Encoder | 3-layer GCN (32, 32, 16), residual on layer 2 | 2-layer, 4-layer, GraphSAGE, GAT, Jumping Knowledge and 4 width shapes all lost |
| Decoder | bilinear + distance, averaged | The ensemble beat either alone on AUPR across all 3 networks — the most robust result found |
| Fusion | continuous averaging | ~0.02 AUROC better than binarise-then-vote on identical models |
| Epochs | 300 | Every decoder variant still improved past 150 |
| Negative sampling | uniform | Hard-negative mining failed catastrophically |
| Iterations | 3 per (algorithm, decoder) | — |

**`docs/REJECTED.md` has the full log of failed experiments.** Read it before
proposing anything — it exists specifically to stop dead ends being relitigated.

### The most important single finding

**Fusion method matters more than model tweaks.** Switching from
binarise-then-count to continuous averaging, with everything else identical,
produced a real gain. Early binarisation throws away exactly how confident each
prediction was.

### Ensemble size is not monotonic

Below ~8 algorithms the ensemble clearly degrades; partial removal down to ~10
is sometimes neutral or slightly better. **This is why a 3-algorithm run looks
bad** — see §7.

---

## 5. Critical gotchas

**Pre-binarised score files.** GENECI's `normalized_weight_matrix_*.csv` files
are already binarised despite the name. Every threshold experiment run on them
was invalid. Always use raw `GRN_<ALGO>.csv`. `loaders.py` now raises if a
confidence column has ≤2 distinct values.

**Matrix orientation.** Canonical on-disk format is **genes × samples**. Some R
packages want the transpose. Each algorithm declares its `Orientation` in
`registry.py`. Getting it wrong yields a plausible-looking, meaningless network.

**DREAM4 `_exp.csv` files are time series, not steady state.** 100 genes × 210
samples = 10 perturbation experiments × 21 time points (t0…t1000, step 50).
Steady-state methods treat the 210 columns as independent conditions (this is
what GENECI does and what the existing validated results used). Time-series
methods need `multigcn/data/dream4.py::to_time_series()`.

**Gold standards are genes × genes matrices**, not edge lists.
`load_gold_standard()` auto-detects both.

**Compute AUROC/AUPR on continuous scores, never post-cutoff.** They measure
ranking across all thresholds; computed after committing to one they always
look worse, and that is a property of the metric, not the method.

**DREAM4 net3 density is 1.94%** (194 true edges), so top_x=0.10 is ~5× true
density. Earlier notes claiming 3.5–5% were wrong.

---

## 6. How to run

```bash
cd MultiGCN
pip install -e ".[dev]"
python -m pytest tests/ -q                  # 38 pass

python examples/run_dream4_net3.py --tier python   # 3 algos, pip only
python examples/run_dream4_net3.py --tier r        # + 12 R algos
python examples/run_dream4_net3.py --tier all
```

```python
from multigcn import MultiGCN

model = MultiGCN(methods=["GENIE3_ET", "CLR", "ARACNE"], top_x=0.10)
print(model.plan())                     # what can actually run here
model.infer("expression.csv", time_series="ts.csv")   # ts optional
model.fit_predict()
model.save_matrices()                   # per-algorithm weight matrices
print(model.per_algorithm_scores("gs.csv"))
```

### Parameters

Namespaced by **package**, not algorithm — GENIE3_RF/ET/GRNBOOST2 share
`genie3_`:

```python
MultiGCN(methods=[...], genie3_ntrees=1000, minet_estimator="pearson",
         bnlearn_R=200, top_x=0.10)     # unprefixed -> Config
```

Validated before anything runs. `model.describe_params()` lists everything.

### Runtimes

| Runtime | N | Needs |
|---|---|---|
| R | 19 | `Rscript` + CRAN/Bioconductor packages |
| Docker | 10 | `docker pull` only — MATLAB tools via free MCR, plus Julia |
| Python | 6 | nothing beyond core deps |
| Isolated | 1 | Docker (NONLINEARODES pins numpy 1.16.3) |

Time-series **required**: DYNGENIE3, NETWORKBMA, SCANBMA, G1DBN, TDARACNE,
SWING, JUMP3, GRNVBEM. Either: DYNUGENE, BNLEARN, GENENET.

**BNLEARN is by far the slowest** — `boot.strength` with R=200 is 200 complete
structure searches. Defaults to cores−1. Time a small `bnlearn_R` first.

---

## 7. Interpreting results

A tier-1 run (3 tree-ensemble algorithms) gives:

```
GENIE3_RF        input  AUROC 0.745  AUPR 0.118
MultiGCN_fused   fused  AUROC 0.685  AUPR 0.092
```

**The fusion losing to its inputs here is expected, not a bug.** Three
algorithms from one family are near-identical; the GCN has nothing to
triangulate and can only lose information. This reproduces the known
ensemble-size finding. Judge the method only with ≥10 algorithms spanning
multiple families.

---

## 8. Open work, roughly prioritised

1. **Run `--tier r` to completion** and confirm the ensemble beats its best
   input. Nothing else matters until this is verified.
2. **DREAM5 / E. coli validation.** All results so far are DREAM4 synthetic.
   Before this: make output sparse at write time (~4,500 genes → 20M rows →
   250 MB files; this is what broke the git push).
3. **DREAM4 regression test** pinning current AUPR so refactors can't silently
   change results.
4. Parallelism across algorithms (`n_jobs` is accepted but ignored).
5. Pull and test the 10 Docker algorithms. PIDC is the most promising —
   partial information decomposition is a distinct family and a strong BEELINE
   performer.
6. Density multiplier for E. coli: does the 2–3× rule transfer?
7. Tukey `k` never validated for the bilinear+distance ensemble output.
8. No held-out validation split yet. Build one before paper-ready numbers.
9. DREAM4 network 2 resisted nearly every improvement that helped 1 and 3 —
   cause never identified.

---

## 9. Working preferences

- Direct, concise, practical. No preamble.
- **Edit files in place** with the filesystem tools. Do not deliver tarballs or
  rewrite whole files when a surgical edit will do.
- Do not create new notebooks unless results beat the baseline.
- Flag uncertainty explicitly — especially unverified library signatures.
- `docs/REJECTED.md` is authoritative on what has already failed.

## 10. Related paths (local machine)

```
~/Documents/Github/GENECI       # source of the 4 remaining vendored scripts
~/Documents/Github/SWING        # SWING clone (swing_source_path)
~/Documents/Github/dynGENIE3    # dynGENIE3 clone (dyngenie3_source_path)
~/Documents/Github/dynUGENE     # dynUGENE R package (devtools::install_local)
```
