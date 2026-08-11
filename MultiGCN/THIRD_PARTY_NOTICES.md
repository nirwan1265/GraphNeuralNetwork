# Third-party notices

## GENECI

MultiGCN vendors the per-algorithm wrapper scripts from **GENECI**
(https://github.com/AdrianSeguraOrtiz/GENECI), Copyright (c) 2022 Adrian Segura
Ortiz, distributed under the MIT License. The MIT License permits
redistribution with attribution; the full text is reproduced in `licenses/GENECI-LICENSE`.

Vendored files live in `multigcn/inference/scripts/` and carry an attribution
header. The only modification is rewriting the `source()` path for the shared R
helpers, which in GENECI resolves against a container WORKDIR that does not
exist outside Docker.

If you use MultiGCN in published work, cite GENECI:

> Segura-Ortiz, A., García-Nieto, J., Aldana-Montes, J.F., Navas-Delgado, I.
> GENECI: A novel evolutionary machine learning consensus-based approach for
> the inference of gene regulatory networks. *Computers in Biology and
> Medicine*, 2023.

## Underlying algorithms

MultiGCN does **not** redistribute the algorithm implementations themselves.
Each is installed separately from its own upstream source under its own
license:

| Algorithm | Implementation | Source |
|---|---|---|
| ARACNE, CLR, MRNET, MRNETB | `minet` | Bioconductor |
| BC3NET | `bc3net` | CRAN |
| C3NET | `c3net` | CRAN |
| PCIT | `CeTF` | Bioconductor |
| KBOOST | `KBoost` | Bioconductor |
| LEAP | `LEAP` | CRAN |
| TIGRESS | `tigress` | github.com/jpvert/tigress |
| GENIE3_RF/ET, GRNBOOST2 | `arboreto` | PyPI |
| INFERELATOR | `inferelator` | PyPI |
| NONLINEARODES | vendored script + pinned deps | see below |
| MEOMI | vendored source tree | GENECI |

Cite the original algorithm papers for whichever methods you use.
