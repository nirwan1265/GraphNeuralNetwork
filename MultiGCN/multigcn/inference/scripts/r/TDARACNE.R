# MultiGCN wrapper for TDARACNE (time-delayed ARACNE).
#
# TIME SERIES REQUIRED. Plain ARACNE computes symmetric mutual information and
# therefore cannot say which gene regulates which. TDARACNE looks at expression
# changes across a time lag, so it can orient edges by which gene moves first -
# a genuinely different signal from the ARACNE already in the pipeline.
#
# Availability warning: TDARACNE has been deprecated in some Bioconductor
# releases. Check it installs before adding it to a benchmark run.

local({
  self <- sub("^--file=", "",
              commandArgs(trailingOnly = FALSE)[grep("^--file=",
                          commandArgs(trailingOnly = FALSE))])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("TDARACNE", "BiocManager::install('TDARACNE')")
RequirePackage("Biobase", "BiocManager::install('Biobase')")
suppressMessages({ library(TDARACNE); library(Biobase) })

if (is.null(args$ts_file))
  stop("TDARACNE requires time-series data. Pass time_series='...csv'.",
       call. = FALSE)

delta      <- if (!is.null(p$delta)) as.integer(p$delta) else 3
likelihood <- if (!is.null(p$likelihood)) as.numeric(p$likelihood) else 1.2
norm       <- if (!is.null(p$norm)) as.integer(p$norm) else 1
thresh     <- if (!is.null(p$thresh)) as.numeric(p$thresh) else 0

raw <- read.csv(args$ts_file, check.names = FALSE)
tcol <- grep("^time$", tolower(names(raw)))
if (length(tcol) == 0)
  stop(paste("Expected a 'time' column in", args$ts_file), call. = FALSE)

# TDARACNE wants an ExpressionSet: genes as rows, time points as columns.
mat <- t(as.matrix(raw[, -tcol[1], drop = FALSE]))
genes <- rownames(mat)
cat("TDARACNE:", length(genes), "genes,", ncol(mat), "time points\n")

if (delta >= ncol(mat))
  stop(sprintf("tdaracne_delta=%d but only %d time points available.",
               delta, ncol(mat)), call. = FALSE)

eset <- ExpressionSet(assayData = mat)
# adj = FALSE returns the weighted network. adj = TRUE returns a binary
# adjacency matrix, which is useless here: the pipeline ranks pairs by
# continuous confidence, so a 0/1 matrix carries no ranking information and
# the loader rejects it outright.
net <- TDARACNE(eset, N = length(genes), delta = delta,
                likehood = likelihood, norm = norm, thresh = thresh,
                ksd = 0, tolerance = 0, adj = FALSE)

# TDARACNE returns a graphNEL (S4), not a matrix, so as.matrix() fails with
# "no method for coercing this S4 class to a vector". Convert via the graph
# package's adjacency-matrix accessor, which preserves edge weights.
if (inherits(net, "graphNEL")) {
  RequirePackage("graph", "BiocManager::install('graph')")
  score <- as(net, "matrix")
} else if (is.list(net) && !is.null(net$net)) {
  score <- as.matrix(net$net)
} else {
  score <- as.matrix(net)
}
dimnames(score) <- list(genes, genes)

conf <- expand.grid(source = genes, target = genes, stringsAsFactors = FALSE)
conf$conf <- as.vector(score)
WriteList(ProcessList(conf), args$output_folder, "TDARACNE")
