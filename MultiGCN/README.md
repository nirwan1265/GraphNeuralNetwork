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
