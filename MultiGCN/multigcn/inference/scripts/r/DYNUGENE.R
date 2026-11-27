# MultiGCN wrapper for dynUGENE.
#
# dynUGENE has two entry points: inferNetwork() for time-series data and
# inferSSNetwork() for steady state. The wrapper picks based on whether a
# time-series file was supplied, so the same algorithm name works for both
# DREAM4 steady-state and time-series runs.

local({
  self <- sub("^--file=", "",
              commandArgs(trailingOnly = FALSE)[grep("^--file=",
                          commandArgs(trailingOnly = FALSE))])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("dynUGENE",
  "devtools::install_local('~/Documents/Github/dynUGENE')")
suppressMessages(library(dynUGENE))

ntree <- if (!is.null(p$ntree)) as.integer(p$ntree) else 10L
mtry  <- if (!is.null(p$mtry) && as.integer(p$mtry) > 0) as.integer(p$mtry) else NULL
alpha <- if (!is.null(p$alpha)) p$alpha else "fromData"
# inferNetwork rejects a non-integer ntree/mtry outright (it type-checks with
# typeof(), so 500 is not 500L).
seed  <- if (!is.null(p$seed)) as.integer(p$seed) else 777L
multi <- if (!is.null(p$multipleExp)) as.logical(p$multipleExp) else FALSE

# alpha may arrive from JSON as a number rather than the string "fromData".
if (!identical(alpha, "fromData")) alpha <- as.numeric(alpha)

if (ntree <= 10)
  cat("note: dynUGENE ntree default is only 10; consider dynugene_ntree=500+\n")

if (!is.null(args$ts_file)) {
  # Time-series mode: first column must be the time stamp.
  raw <- read.csv(args$ts_file, check.names = FALSE)
  tcol <- grep("^time$", tolower(names(raw)))
  if (length(tcol) == 0)
    stop(paste("Expected a 'time' column in", args$ts_file), call. = FALSE)

  # Auto-detect concatenated experiments. dynUGENE models dExpression/dt, so if
  # several experiments are laid end to end and it is told there is only one,
  # it computes a derivative across each boundary where time jumps backwards
  # (e.g. 1000 -> 0). That divides by a negative or zero dt and yields NaN/Inf,
  # surfacing much later as randomForest's opaque
  #   "NA/NaN/Inf in foreign function call (arg 2)".
  # Detecting it here is strictly better than relying on the caller to set a
  # flag correctly.
  times <- raw[[tcol[1]]]
  n_experiments <- 1L + sum(times[-1] <= times[-length(times)])
  if (n_experiments > 1L && !multi) {
    cat("dynUGENE: detected", n_experiments,
        "concatenated experiments; enabling multipleExp\n")
    multi <- TRUE
  }

  cat("dynUGENE: time-series mode,", nrow(raw), "measurements,",
      n_experiments, "experiment(s)\n")

  # dynUGENE's estimateDecayRates computes
  #     alpha = (log(xmax) - log(xmin)) / abs(tmin - tmax)
  # where tmin/tmax are TIME VALUES, not row indices. With concatenated
  # experiments a gene's minimum and maximum can fall at the same timestamp in
  # two different experiments, making the denominator zero and alpha Inf. That
  # Inf flows into `decays` and surfaces as randomForest's opaque
  #     "NA/NaN/Inf in foreign function call (arg 2)".
  #
  # A repaired VECTOR cannot be passed back: inferNetwork validates with
  #     if (length(alpha) != ngenes) stop(...)
  # where ngenes = dim(data)[2] still counts the time column, so it demands
  # 101 entries for 100 genes. No vector can satisfy it. A single number takes
  # the working code path (rep(alpha, ngenes - 1)), so collapse to the median
  # of the finite rates - a representative decay rate for this dataset.
  if (identical(alpha, "fromData") && n_experiments > 1L) {
    alphas <- estimateDecayRates(raw)
    finite <- alphas[is.finite(alphas) & alphas > 0]
    repaired <- if (length(finite)) stats::median(finite) else 0.01
    cat("  alpha: using median decay rate", signif(repaired, 3),
        sprintf("(%d of %d gene estimates were non-finite)\n",
                sum(!is.finite(alphas)), length(alphas)))
    alpha <- as.numeric(repaired)
  }

  ugene <- inferNetwork(data = raw, multipleExp = multi, ntree = ntree,
                        mtry = mtry, alpha = alpha, seed = seed,
                        showPlot = FALSE)
} else {
  ex <- as.data.frame(ReadExpression(args$in_file, samples_as_rows = TRUE))
  cat("dynUGENE: steady-state mode,", nrow(ex), "conditions\n")
  ugene <- inferSSNetwork(data = ex, ntree = ntree, mtry = mtry,
                          seed = seed, showPlot = FALSE)
}

net <- ugene$network
genes <- colnames(net)
if (is.null(genes)) genes <- rownames(net)

conf <- expand.grid(source = genes, target = genes, stringsAsFactors = FALSE)
conf$conf <- as.vector(net)
WriteList(ProcessList(conf), args$output_folder, "DYNUGENE")
