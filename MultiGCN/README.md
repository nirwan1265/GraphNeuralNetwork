# MultiGCN

Gene regulatory network reconstruction by combining many inference algorithms
through a graph convolutional autoencoder.

Every GRN inference algorithm captures a different partial view of the
regulatory structure, with its own blind spots. MultiGCN runs many of them,
learns a graph representation from each, and fuses the results into a single
continuous network — recovering edges that no individual algorithm proposed.

**36 algorithms across 10 methodological families**, behind one interface. No
dependency on any external orchestration tool.

```python
from multigcn import MultiGCN

model = MultiGCN(methods=["GENIE3_ET", "CLR", "ARACNE", "PPCOR"], top_x=0.10)
model.infer("expression.csv")
scores = model.fit_predict()
print(model.evaluate("goldstandard.csv"))
```

---

## The model: a joint multi-relational graph autoencoder

The core of MultiGCN is a **graph autoencoder (GAE)**: it compresses every gene
into a short vector that summarises its position in the network, then rebuilds
the network from those vectors. Pairs of genes that the rebuilt network scores
highly but the input lacked are the model's predicted new edges.

**One model, many graphs.** Every inference algorithm gives its own graph over
the same genes. MultiGCN does not train one model per graph. It puts them all
into a single graph with one *edge type per algorithm* and trains one model on
it. Each gene gets **one** vector that has to explain every algorithm's graph at
once, so the model is forced to keep the structure the algorithms share - which
is where the new edges come from.

**Input graphs.** Each node is a gene. For each algorithm, edges are its own
strongest gene-gene pairs (top 10% of pairs, ranked within that algorithm, so
algorithms on different score scales get graphs of the same size). Every gene
starts with a one-hot identity vector as its feature, so the model learns from
network structure alone.

**Encoder.** Two relational graph convolution layers (R-GCN). In each one,
every gene updates its vector from its neighbours, with separate weights for
each edge type. Those weights are built from 8 shared *bases*, so the edge
types share parameters instead of each learning in isolation.

| Layer | Size | Extras |
|---|---|---|
| RGCNConv 1 | n genes -> 32 | ReLU, dropout |
| RGCNConv 2 | 32 -> 16 | - |

**Embeddings.** The encoder's output is a 16-number vector `z` for every gene.

**Decoder.** One DistMult decoder per algorithm scores every gene pair:
`s_r(i, j) = sigma(sum_k z_ik * R_rk * z_jk)`, where `R_r` is a learned 16-number
vector for algorithm `r`. The gene vectors are shared; `R_r` captures how
algorithm `r` reads them.

**Training.** For every algorithm, its edges are positives and an equal number
of randomly drawn gene pairs (redrawn each epoch) are negatives; binary
cross-entropy, averaged over algorithms. Adam, learning rate 0.01, 300 epochs,
3 random seeds averaged. Do not train longer or widen the embedding: both lower
the training loss and lower accuracy, because a model that can rebuild every
graph exactly has memorised them instead of keeping what they share.

**Output.** Each algorithm's reconstruction is scaled to [0, 1] and combined,
weighting each algorithm by how well it agrees with the others (no gold
standard used). That is blended 50/50, by rank, with the rank-average of the
algorithms' raw scores: the model is strongest at the top of the ranking, the
raw-score average across the whole ranking. The result is a continuous score
for every gene pair; a cutoff is applied only if a binary network is needed.

**On DREAM4** (5 networks of 100 genes, 23 algorithms):

| | AUPR | AUROC |
|---|---|---|
| Rank-average of raw scores, no model | 0.149 | 0.719 |
| One GCN autoencoder per algorithm (previous model) | 0.159 | 0.694 |
| **Joint model (default)** | **0.171** | **0.722** |

The joint model beat the per-algorithm model on all five networks, including
two held out from tuning. The previous model is still available as
`model="separate"`; its 3-layer GCN encoder and bilinear + distance decoders
are documented in `multigcn/models/`.

```python
MultiGCN(model="joint")              # default
MultiGCN(model="separate", fusion_weights="flat", input_blend=0)   # previous
```

See [`docs/REJECTED.md`](docs/REJECTED.md) for every alternative that was
tested and did worse.

---

## Install

```bash
pip install -e .            # core pipeline
pip install -e ".[dev]"     # + test suite
```

`torch-geometric` must match your torch/CUDA build — see the
[official install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/).

Six algorithms run immediately with no further setup. The rest need their own
runtime:

| Runtime | Count | Requirement |
|---|---|---|
| Python | 6 | nothing beyond core dependencies |
| R | 19 | `Rscript` on PATH + the relevant CRAN/Bioconductor packages |
| Docker | 10 | `docker pull` only |
| Isolated | 1 | Docker (dependencies cannot share the interpreter) |

`multigcn plan --all` reports exactly which are runnable on your machine before
you start a long job.

### R packages

```r
BiocManager::install(c("minet", "CeTF", "KBoost", "networkBMA"))
install.packages(c("bc3net", "c3net", "LEAP", "ppcor", "GeneNet",
                   "G1DBN", "bnlearn"))
```

### Docker algorithms

JUMP3, NARROMI, GRNVBEM, PLSNET, CMI2NI, PCACMI, LOCPCACMI and RSNET are MATLAB
tools; PIDC and PUC are Julia. **You do not need MATLAB or Julia** — they ship
as prebuilt images compiled against the free MATLAB Compiler Runtime:

```bash
docker pull adriansegura99/geneci_infer-network_pidc:4.0.0
```

MultiGCN pulls on demand. On Apple Silicon these are `linux/amd64` images and
run under emulation — slower, but they work.

---

## Usage

```python
model = MultiGCN(
    methods=["GENIE3_RF", "GENIE3_ET", "CLR", "ARACNE", "MRNET", "PPCOR"],
    top_x=0.10,
    epochs=300,
)

print(model.plan())          # what will run, and whether it can

model.infer("expression.csv", skip_failures=True)
scores = model.fit_predict()

model.save()                 # fused network
model.save_matrices()        # per-algorithm weight matrices
print(model.per_algorithm_scores("goldstandard.csv"))
```

Everything at once:

```python
model = MultiGCN(all_methods=True)
```

Already have inferred networks? Skip inference entirely:

```python
model.load("path/to/lists", expression_file="expression.csv")
model.fit_predict()
```

CLI:

```bash
multigcn list                    # the algorithm catalogue
multigcn plan --all              # what can run here
multigcn run expression.csv --all -c config.yaml --gold-standard gs.csv
```

### Parameters

Algorithm parameters are flat and namespaced by the originating **package**:

```python
MultiGCN(
    methods=["GENIE3_ET", "ARACNE", "BNLEARN"],
    genie3_ntrees=1000,          # -> tree ensemble size
    minet_estimator="pearson",   # -> shared by ARACNE/CLR/MRNET/MRNETB
    bnlearn_algorithm="tabu",
    top_x=0.10,                  # unprefixed -> pipeline config
)
```

Everything is validated before any algorithm runs, so a typo or out-of-range
value fails immediately rather than being silently ignored.
`model.describe_params()` lists every tunable for your selection.

---

## Input formats

**Expression** — genes × samples CSV, gene ids in column 1, sample ids in the
header. Each algorithm declares the orientation its library expects and the
wrapper transposes as needed.

**Time series** — first column `time`, remaining columns genes. A time value
that does not increase marks the start of a new experiment. Required by
DYNGENIE3, NETWORKBMA, SCANBMA, G1DBN, TDARACNE, SWING, JUMP3 and GRNVBEM,
which model expression *change* and cannot run on steady-state data:

```python
model.infer("expression.csv", time_series="timeseries.csv")
```

DREAM4 users: `multigcn.data.dream4.to_time_series()` converts the standard
`*_exp.csv` layout.

**Gold standard** — genes × genes matrix or 3-column edge list. Auto-detected.

**Edge lists** — headerless `source,target,confidence` with *continuous*
scores. The loader rejects files whose confidence column has ≤2 distinct
values, since a pre-binarised file silently invalidates any threshold work
downstream.

---

## Adding an algorithm

One entry in `multigcn/registry.py`. Nothing else changes:

```python
register(AlgorithmSpec(
    name="MYMETHOD",
    runtime=Runtime.PYTHON,
    script="python/MYMETHOD.py",
    library="mypackage",
    install="pip install mypackage",
    orientation=Orientation.GENES_ROWS,
    family="regression",
    param_prefix="mymethod",
    params={"ntrees": ParamSpec("ntrees", 500, int, minimum=1)},
    vendored=False,
))
```

The wrapper must write `GRN_MYMETHOD.csv` — headerless, 3 columns, continuous
scores — into the output folder given as its second argument.

---

## Notes on results

MultiGCN is an *ensemble* method, and ensemble size is not monotonic: below
roughly eight algorithms the fusion degrades, because near-identical inputs
give the model nothing to triangulate. Running three related algorithms and
concluding the method underperforms is a common mistake. Use a spread of
families.

Report AUROC and AUPR on the continuous output, before any cutoff.

---

## Future work: transcription factor (TF) information

For datasets that come with a TF list — the DREAM5 networks do — every true
regulatory edge starts at a TF. In DREAM5 *E. coli* (network 3), 334 of 4,511
genes are TFs and all 141 regulators in the gold standard are on the list, which
shrinks the candidate pairs from ~20 million to ~1.5 million. That makes the TF
list a hard constraint worth using at two levels.

### Level 1 — in each algorithm

- **Natively supports a regulator list:** GENIE3 / GENIE3_ET / GRNBOOST2 (our
  code), DYNGENIE3, TIGRESS, INFERELATOR, plus our ELASTICNET, PLSNET and
  NONLINEARODES — easy, just restrict the predictors. KBOOST and SWING may
  support it; to be checked.
- **No concept of a regulator** (ARACNE, CLR, MRNET, PEARSON, WGCNA, PIDC,
  PUC): these are symmetric, so just keep the TF → gene rows afterwards.

### Level 2 — in the GCN

Three options, ranked by how easy they are to explain:

1. **Constrain training and output to TF → gene pairs.** Right now negative
   sampling draws random pairs, most of which are gene → gene pairs that are
   trivially false in DREAM5 — so the model learns from easy, uninformative
   negatives. Drawing negatives only from TF → gene pairs makes the training
   objective match the actual task. This is the most promising option, and it
   is a one-sentence explanation in a methods section.
2. **Directionality comes for free.** The GCN graph is undirected, which is a
   known weakness. With a TF list, any edge between a TF and a non-TF must point
   from the TF. Only TF ↔ TF pairs stay ambiguous.
3. **A TF flag as a node feature.** The model learns *something* about TFs, but
   it is hard to say what. Keep as an ablation only.

**Planned test:** a 2×2 ablation on DREAM5 — TF information at inference
(yes / no) × in the GCN (yes / no) — to show where the gain comes from. Planned
API: `MultiGCN(methods=[...], regulators="transcription_factors.tsv")`.
Use the TF *list* only; feeding known edges (e.g. from RegulonDB) into the model
would leak the answer into the evaluation.

---

## Attribution

Several R wrappers are adapted from
[GENECI](https://github.com/AdrianSeguraOrtiz/GENECI) (MIT, © 2022 Adrián
Segura Ortiz), and the Docker images for the MATLAB and Julia algorithms are
GENECI's. Algorithm implementations themselves are installed from their own
upstream sources under their own licenses and are **not** redistributed here.
See `THIRD_PARTY_NOTICES.md`, and cite the original algorithm papers for
whichever methods you use.

## License

MIT — see `LICENSE`.
