# MultiGCN

Reconstruct gene regulatory networks by combining many inference algorithms
through a GCN graph autoencoder, rather than trusting any single algorithm.

Each algorithm captures a different partial view of the regulatory structure
with its own blind spots. MultiGCN trains a separate GCN encoder per algorithm,
then fuses all reconstructions into one continuous network — recovering edges
no individual algorithm proposed.

## Install

```bash
pip install -e .                 # core (GCN pipeline)
pip install -e ".[inference]"    # + Python-runtime algorithms
pip install -e ".[dev]"          # + test suite
```

`torch-geometric` must match your torch/CUDA build — install it from the
[official instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/).

### Vendoring the wrapper scripts

The per-algorithm wrappers come from GENECI (MIT). Populate them once from a
local checkout:

```bash
python -m multigcn.inference.vendor --geneci-path ~/Documents/Github/GENECI
```

### Algorithm runtimes

The 16 algorithms are not all Python, and this is a hard constraint rather than
a design choice:

| Runtime | Count | Requirement |
|---|---|---|
| R | 11 | `Rscript` on PATH + the relevant CRAN/Bioconductor packages |
| Python (in-process) | 4 | `pip install ".[inference]"` |
| Isolated | 1 | Docker — NONLINEARODES pins numpy 1.16.3 / Python 3.6 and **cannot** share an interpreter with PyTorch |

`multigcn plan` reports exactly which are runnable on your machine before you
start a long job.

## Usage

```python
from multigcn import MultiGCN

model = MultiGCN(methods=["GENIE3_ET", "CLR", "ARACNE"], top_x=0.10)
print(model.plan())

model.infer("expression.csv")
scores = model.fit_predict()
print(model.evaluate("goldstandard.tsv", k=1.5))
```

Everything at once:

```python
model = MultiGCN(all_methods=True, epochs=300, iterations=3)
```

Already have inferred networks? Skip inference entirely:

```python
model.load("path/to/lists", expression_file="expression.csv")
model.fit_predict()
```

CLI:

```bash
multigcn list                                   # catalogue
multigcn plan --all                             # what can run here
multigcn run expression.csv --all -c config.yaml --gold-standard gs.tsv
```

## Input format

Expression: **genes × samples** CSV, gene ids in column 1, sample ids in the
header. The wrappers transpose internally where their library needs it — the
orientation each one expects is recorded per-algorithm in `registry.py`, since
getting it wrong yields a plausible-looking but meaningless network.

Edge lists: headerless `source,target,confidence` with *continuous* scores.
The loader rejects files whose confidence column has ≤2 distinct values, because
a pre-binarised file silently invalidates every threshold experiment run on it.

## Adding an algorithm

One `AlgorithmSpec` entry in `multigcn/registry.py`. Nothing else changes:

```python
register(AlgorithmSpec(
    name="MYMETHOD",
    runtime=Runtime.PYTHON,
    script="python/MYMETHOD.py",
    library="mypackage",
    install="pip install mypackage",
    orientation=Orientation.GENES_ROWS,
    family="regression",
))
```

The wrapper must write `GRN_MYMETHOD.csv` — headerless, 3 columns, continuous
scores — into the output folder given as its second argument.

## Design decisions

These are validated on DREAM4, not guesses. Changing one means re-running the
benchmark.

| Component | Setting | Why |
|---|---|---|
| Input topology | top 10%, rank-based | Fixed cutoffs give 0.3%–67% density across algorithms at the same value |
| Node features | identity matrix | Real expression as features tested, clearly worse |
| Encoder | 3-layer GCN (32, 32, 16) | 2/4-layer, GraphSAGE, GAT, JK, 4 width shapes all lost |
| Decoder | bilinear + distance, averaged | Ensemble beat both individually on AUPR, all 3 networks |
| Fusion | continuous averaging | ~0.02 AUROC over binarise-then-count, controlled test |
| Epochs | 300 | Every decoder variant still gained past 150 |
| Negative sampling | uniform | Hard-negative mining failed catastrophically |

`docs/REJECTED.md` has the full log of what was tried and did not work — read it
before proposing a change.

## Attribution

Wrapper scripts vendored from [GENECI](https://github.com/AdrianSeguraOrtiz/GENECI)
(MIT, © 2022 Adrián Segura Ortiz). Algorithm implementations are installed from
their own upstream sources under their own licenses and are **not** redistributed
here. See `THIRD_PARTY_NOTICES.md`.
