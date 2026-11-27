# MultiGCN — project handoff

**Read this first if you are a new assistant session picking up this work.**
Then read `docs/REJECTED.md` before proposing any change to the method.

> ⚠️ **Contains unpublished methodology.** The combination approach is novel and
> not yet published. Keep it out of public material until the paper is out.
> `README.md` is deliberately vaguer than this file — that is intentional.

---

## 1. What this is

Two goals, one codebase:

1. **A publishable method** — a GCN graph-autoencoder ensemble for gene
   regulatory network (GRN) inference that beats existing ensemble methods.
2. **A pip-installable package** (`MultiGCN`) with no dependency on GENECI or
   any other external orchestration tool.

Owner: Nirwan Tandukar, PhD student, computational biology. Also applying to
industry roles, so repo presentation matters.

### The method

1. Run **N inference algorithms** on the same expression data; each gives a
   continuous confidence for every gene pair.
2. Per algorithm, keep its **top 10% of pairs** as a graph (rank-based, not an
   absolute cutoff).
3. Train a **separate GCN autoencoder** on each graph. Node features are the
   identity matrix — no expression values.
4. Decode with **two decoders** (bilinear + distance), **3 iterations** each.
5. **Average all reconstructions continuously** into one network.

---

## 2. Current state

### Working
- 38 tests pass. `pip install -e ".[dev]"`.
- **36 algorithms registered**, 10 families. **23–25 actually runnable** (§4).
- Full pipeline verified end to end on DREAM4 100-gene network 3.
- GENIE3/GRNBOOST2 reimplemented natively on scikit-learn (no arboreto).
  Verified: `GENIE3_RF` ntrees=1000 → AUROC 0.742 / AUPR 0.125.
- **All 9 GENECI R wrappers rewritten as MultiGCN's own code.** `vendor.py` is
  no longer needed for anything.
- **One CLI.** Four overlapping example scripts were consolidated into
  `multigcn/cli.py`.

### The open problem — read before "fixing" anything
On the full 18-algorithm run, **the fused network scored below its best single
input**: fused AUPR 0.135 vs SWING 0.244. **Every one of 34 reconstructions
scored below its own input.** A topology sweep (`top_x` ∈ {0.02, 0.03, 0.05,
0.10} × `keep_weights` ∈ {False, True}) moved fused AUPR only between 0.125 and
0.135 — so topology density is *not* the cause.

Two live hypotheses:

1. **Flat averaging dilutes a strong outlier.** SWING is 2× the next best; the
   mean of 18 drags it toward the pack. Test: fuse only 6–8 strong,
   family-diverse algorithms.
2. **`keep_weights` does nothing — confirmed bug.** `edge_index_from_adj()`
   calls `np.nonzero(adj)` and discards weights; `train_gae` never passes
   `edge_weight` to `GCNConv`. So the GCN *always* sees a binary graph and the
   option cannot have an effect. This is why both sweep columns were identical
   to six decimals. **Not yet fixed** — it changes validated architecture, so
   ask before touching.

### Not verified
- Docker algorithms have never run (see §4).
- Several R argument names written from memory (`c3net` alpha/cutoffMI,
  `kboost` ite, `TDARACNE`). The `AddIfSet` helper only forwards a parameter
  when explicitly set, so defaults are safe.

---

## 3. Usage — one command

```bash
multigcn list                                     # algorithms + keywords
multigcn run <exp.csv> --algorithms all --dry-run # show plan, don't run
multigcn run <exp.csv> --algorithms all -g <gs.csv>
multigcn run <exp.csv> --algorithms all --no-fuse # inference only
multigcn run <exp.csv> --algorithms cached --use-cached -g <gs.csv>
```

`--algorithms` takes keywords, explicit names, or a mix (`r,SWING,PIDC`). All
keyword groups are **derived from the registry**, never hardcoded:

| keyword | n | meaning |
|---|---|---|
| `all` | 25 | everything not needing Docker |
| `everything` | 36 | including Docker |
| `r` / `python` / `docker` | 19 / 6 / 11 | by runtime |
| `timeseries` / `steadystate` | 8 / 27 | by data requirement |
| `cached` | varies | whatever has an edge list already |
| family name | varies | `tree_ensemble`, `bayesian`, … |

Parameters are `--<package>-<param>`, prefixed by **package** not algorithm
(GENIE3_RF/ET/GRNBOOST2 share `genie3_`):

```bash
multigcn run <exp.csv> --algorithms all --genie3-ntrees 500 --bnlearn-R 20
```

Everything is validated before any algorithm runs.

Python API mirrors this: `MultiGCN(methods=[...], genie3_ntrees=500,
top_x=0.10)` — unprefixed kwargs go to `Config`.

---

## 4. What runs and what doesn't

**23 working** (edge lists exist in `dream4_net3_out/lists/`):
ARACNE, BC3NET, BNLEARN, C3NET, CLR, DYNGENIE3, G1DBN, GENENET, GENIE3_ET,
GENIE3_RF, GRNBOOST2, INFERELATOR, KBOOST, LEAP, MEOMI, MRNET, MRNETB,
NETWORKBMA, NONLINEARODES, PCIT, PPCOR, SWING, TIGRESS

**Pending**: DYNUGENE (fix applied, needs retest), TDARACNE (`BiocManager::install('TDARACNE')`).

**Blocked — environment, not bugs. Do not try to "fix" these:**

- **10 Docker algorithms** (JUMP3, NARROMI, GRNVBEM, PLSNET, CMI2NI, PCACMI,
  LOCPCACMI, RSNET, PIDC, PUC). GENECI's images fail with
  `cannot enable executable stack as shared object requires`. Old shared
  libraries request an executable stack; kernels ≥5.8 refuse. Affects Julia and
  MATLAB images alike. They *used* to work on this Mac — a Docker Desktop
  update broke them. Untried: setting `useVirtualizationFrameworkRosetta:false`
  in `~/Library/Group Containers/group.com.docker/settings.json`.
- **SCANBMA** — needs networkBMA 2.x, but networkBMA was **removed from
  Bioconductor**. Only the archived 1.0.0 is obtainable and it has no ScanBMA.
  NETWORKBMA works on 1.0.0 and is unaffected.

**Recovering blocked algorithms:** GENECI's historical output is on disk and
importable — this is how MEOMI, NONLINEARODES, INFERELATOR and TIGRESS became
available despite Docker being broken:

```bash
python examples/import_lists.py \
  --from ../data/inferred_networks/dream4_100_03_exp/lists \
  --output dream4_net3_out
```

---

## 5. Validated design — do not change casually

| Component | Setting | Why |
|---|---|---|
| Input topology | top 10%, rank-based | fixed cutoffs gave 0.3%–67% density across algorithms |
| Node features | identity matrix | expression-as-features tested, clearly worse |
| Encoder | 3-layer GCN (32,32,16), residual on layer 2 | 2/4-layer, GraphSAGE, GAT, JK, 4 width shapes all lost |
| Decoder | bilinear + distance, averaged | beat either alone on AUPR across all 3 networks |
| Fusion | continuous averaging | ~0.02 AUROC over binarise-then-vote |
| Epochs | 300 | all variants still improving past 150 |
| Negative sampling | uniform | hard-negative mining failed catastrophically |

**Biggest single finding: fusion method matters more than model tweaks.**
Binarising early destroys confidence information.

**Ensemble size is not monotonic.** Below ~8 algorithms it clearly degrades.
A 3-algorithm run scoring badly is expected, not a bug.

---

## 6. Gotchas that cost real time

- **Pre-binarised files.** GENECI's `normalized_weight_matrix_*.csv` are binary
  despite the name; every experiment run on them was invalid. Use raw
  `GRN_*.csv`. The loader now raises on ≤2 distinct scores.
- **Matrix orientation.** Canonical format is **genes × samples**; each
  algorithm declares its `Orientation`. Wrong orientation gives a
  plausible-looking meaningless network.
- **DREAM4 `_exp.csv` are time series, not steady state.** 100 genes × 210
  samples = 10 perturbation experiments × 21 time points (t0…t1000, step 50).
  `multigcn/data/dream4.py::to_time_series()` reshapes; the CLI does it
  automatically when a selected algorithm needs it.
- **Gold standards are genes × genes matrices**, not edge lists. Auto-detected.
- **AUROC/AUPR on continuous scores only**, never post-cutoff.
- **DREAM4 net3 density is 1.94%** (194 edges). Earlier notes saying 3.5–5%
  were wrong, so `top_x=0.10` is ~5× true density.
- **R runners execute with `cwd` set to the scripts dir**, so any path passed
  to them must be absolute.
- **Two upstream bugs found in dynUGENE**, both worked around in the wrapper:
  `estimateDecayRates` divides by `abs(tmin - tmax)` using *time values*, which
  is zero when a gene's min and max share a timestamp across experiments →
  `alpha = Inf`; and `inferNetwork` validates `length(alpha) != ngenes` where
  `ngenes` still counts the time column, so a vector alpha can never pass.

---

## 7. Layout

```
GraphNeuralNetwork/                  <- outer repo (research, data, papers)
├── data/Database.DREAM4/EXP|GS/     <- expression / gold standards
├── data/inferred_networks/          <- GENECI's historical output (importable)
├── scripts/pre-commit               <- blocks >50 MB commits
└── MultiGCN/                        <- the package (NOT a separate git repo)
    ├── multigcn/
    │   ├── registry.py              <- SINGLE SOURCE OF TRUTH, all 36 algorithms
    │   ├── cli.py                   <- THE entry point
    │   ├── params.py                <- <package>_<param> routing + validation
    │   ├── pipeline.py              <- MultiGCN class
    │   ├── config.py                <- validated defaults
    │   ├── inference/
    │   │   ├── runners.py           <- R / Python / Docker (auto-pulls images)
    │   │   └── scripts/r|python/    <- 20 wrappers, all MultiGCN-owned
    │   ├── data/dream4.py           <- time-series conversion
    │   ├── topology/threshold.py    <- rank-based top-X%
    │   ├── models/                  <- encoders.py, decoders.py
    │   ├── training/train.py
    │   ├── fusion/combine.py
    │   └── evaluation/metrics.py
    ├── docs/REJECTED.md             <- READ BEFORE PROPOSING CHANGES
    ├── docs/HANDOFF.md              <- this file
    ├── examples/import_lists.py     <- import precomputed edge lists
    └── tests/                       <- 38 tests
```

---

## 8. Next steps, prioritised

1. **Resolve the fusion problem** (§2). Test the subset hypothesis first — it
   needs no code change: `--algorithms SWING,DYNGENIE3,GENIE3_RF,CLR,MRNETB,PCIT
   --use-cached`. Then decide on the `edge_weight` bug.
2. Retest DYNUGENE; install TDARACNE.
3. **DREAM4 regression test** pinning current AUPR so refactors can't silently
   change results.
4. **DREAM5 / E. coli.** First make output sparse at write time — 4,500 genes →
   20M rows → 250 MB files, which is what broke a git push.
5. Cross-check `to_time_series()` against networkBMA's built-in `dream4ts100`.
   Eight algorithms depend on that reshape and it is only self-verified.
6. Parallelism across algorithms (`n_jobs` accepted but ignored).
7. No held-out validation split yet. Build one before paper-ready numbers.
8. DREAM4 network 2 resisted every improvement that helped 1 and 3 — unexplained.

---

## 9. Working preferences

- Direct and concise. No preamble.
- **Edit files in place** with the filesystem tools. Do not deliver tarballs or
  rewrite whole files when a surgical edit will do.
- **Do not add new scripts.** Everything belongs in `cli.py` behind one
  command. Four overlapping example scripts were already deleted once for this.
- Flag uncertainty explicitly, especially unverified library signatures.
- `docs/REJECTED.md` is authoritative on what has already failed.

## 10. Local paths

```
~/Documents/Github/GENECI       # reference only; no longer a dependency
~/Documents/Github/SWING        # --swing-source-path
~/Documents/Github/dynGENIE3    # --dyngenie3-source-path
~/Documents/Github/dynUGENE     # devtools::install_local
```
