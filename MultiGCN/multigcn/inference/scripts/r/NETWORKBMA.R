# MultiGCN wrapper for networkBMA (Bioconductor).
#
# networkBMA fits, for each gene, a linear model predicting its expression at
# time t+1 from all genes at time t, using Bayesian Model Averaging. It is
# inherently a time-series method: without ordered time points there is no
# "next" measurement to predict, so steady-state data cannot be used.
#
# Output is already (regulator, target, posterior probability) with probability
# in [0,1] - exactly the pipeline's format, so no rescaling is required.

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

# Time-series CSV: first column = time point, remaining columns = genes.
raw <- read.csv(ts_path, check.names = FALSE)
tcol <- grep("^time$", tolower(names(raw)))
if (length(tcol) == 0)
  stop(paste("networkBMA needs time-series data: expected a 'time' column in",
             ts_path, "\nPass one with --time-series <file>."), call. = FALSE)

nTimePoints <- length(unique(raw[[tcol[1]]]))
expr <- raw[, -tcol[1], drop = FALSE]
if (nTimePoints < 3)
  stop(sprintf("Only %d distinct time points; networkBMA needs several.",
               nTimePoints), call. = FALSE)

nvar     <- if (!is.null(p$nvar)) as.integer(p$nvar) else 50
ordering <- if (!is.null(p$ordering)) p$ordering else "bic1"
self_ed  <- if (!is.null(p$self)) as.logical(p$self) else TRUE
thres    <- if (!is.null(p$thresProbne0)) as.integer(p$thresProbne0) else 5

# 'bic1+prior' orderings require prior.prob, which we do not supply.
if (grepl("prior", ordering) ) {
  warning("ordering requests a prior but none is supplied; falling back to 'bic1'.")
  ordering <- "bic1"
}

cat("networkBMA:", nTimePoints, "time points,", ncol(expr), "genes\n")
edges <- networkBMA(
  data = expr, nTimePoints = nTimePoints, ordering = ordering,
  nvar = min(nvar, ncol(expr)), self = self_ed,
  control = iBMAcontrolLM(thresProbne0 = thres)
)

conf <- data.frame(
  source = as.character(edges[[1]]),
  target = as.character(edges[[2]]),
  conf   = as.numeric(edges[[3]]),
  stringsAsFactors = FALSE
)

# Probabilities are already on a meaningful [0,1] scale; only drop zeros and
# self-edges rather than min-max rescaling away the absolute interpretation.
conf <- conf[conf$source != conf$target & conf$conf > 0, ]
conf <- conf[order(conf$conf, decreasing = TRUE), ]
WriteList(conf, args$output_folder, "NETWORKBMA")
