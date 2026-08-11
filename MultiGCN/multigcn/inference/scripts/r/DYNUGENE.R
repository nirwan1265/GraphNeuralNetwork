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
seed  <- if (!is.null(p$seed)) as.integer(p$seed) else 777L
multi <- if (!is.null(p$multipleExp)) as.logical(p$multipleExp) else FALSE

if (ntree <= 10)
  cat("note: dynUGENE ntree default is only 10; consider dynugene_ntree=500+\n")

if (!is.null(args$ts_file)) {
  # Time-series mode: first column must be the time stamp.
  raw <- read.csv(args$ts_file, check.names = FALSE)
  cat("dynUGENE: time-series mode,", nrow(raw), "measurements\n")
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
