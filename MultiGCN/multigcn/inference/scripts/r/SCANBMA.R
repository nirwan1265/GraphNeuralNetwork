# MultiGCN wrapper for ScanBMA (networkBMA package).
#
# TIME SERIES REQUIRED. Same package as NETWORKBMA, so this costs no extra
# install, but a different search strategy: ScanBMA explores model space
# directly with a Zellner g-prior, where the iterative-BMA path used by
# NETWORKBMA prunes greedily. They disagree often enough to be worth running
# both - which is the entire premise of this pipeline.
#
# Output is posterior probability in [0,1], already the format we want.

local({
  self <- sub("^--file=", "",
              commandArgs(trailingOnly = FALSE)[grep("^--file=",
                          commandArgs(trailingOnly = FALSE))])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("networkBMA", "BiocManager::install('networkBMA')")
suppressMessages(library(networkBMA))

ts_path <- if (!is.null(args$ts_file)) args$ts_file else args$in_file
raw <- read.csv(ts_path, check.names = FALSE)
tcol <- grep("^time$", tolower(names(raw)))
if (length(tcol) == 0)
  stop(paste("ScanBMA requires time-series data: no 'time' column in", ts_path,
             "\nPass one with time_series='...csv'."), call. = FALSE)

nTimePoints <- length(unique(raw[[tcol[1]]]))
expr <- raw[, -tcol[1], drop = FALSE]
if (nTimePoints < 3)
  stop(sprintf("Only %d distinct time points; ScanBMA needs several.",
               nTimePoints), call. = FALSE)

g     <- if (!is.null(p$g)) as.numeric(p$g) else 20
oddsR <- if (!is.null(p$odds_ratio)) as.numeric(p$odds_ratio) else 10000
nvar  <- if (!is.null(p$nvar)) as.integer(p$nvar) else 50

cat("ScanBMA:", nTimePoints, "time points,", ncol(expr), "genes\n")

edges <- networkBMA(
  data = expr, nTimePoints = nTimePoints, ordering = "bic1",
  nvar = min(nvar, ncol(expr)), self = TRUE,
  control = ScanBMAcontrol(gCtrl = g, OR = oddsR)
)

conf <- data.frame(
  source = as.character(edges[[1]]),
  target = as.character(edges[[2]]),
  conf   = as.numeric(edges[[3]]),
  stringsAsFactors = FALSE
)
# Posterior probabilities are already meaningfully scaled; drop zeros and
# self-edges rather than min-max rescaling away that interpretation.
conf <- conf[conf$source != conf$target & conf$conf > 0, ]
conf <- conf[order(conf$conf, decreasing = TRUE), ]
WriteList(conf, args$output_folder, "SCANBMA")
