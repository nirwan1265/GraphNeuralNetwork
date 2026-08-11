# MultiGCN wrapper for GeneNet (graphical Gaussian models).
#
# Same goal as ppcor - partial correlations that strip out indirect edges - but
# GeneNet uses a shrinkage estimator, so it remains well defined when genes
# outnumber samples. That is the normal regime for GRN data, which makes this
# the safer of the two partial-correlation methods here.
#
# statistic="prob" returns the posterior probability that an edge is non-zero,
# which is already a calibrated [0,1] confidence.

local({
  self <- sub("^--file=", "",
              commandArgs(trailingOnly = FALSE)[grep("^--file=",
                          commandArgs(trailingOnly = FALSE))])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("GeneNet", "install.packages('GeneNet')")
suppressMessages(library(GeneNet))

method    <- if (!is.null(p$method)) p$method else "static"
statistic <- if (!is.null(p$statistic)) p$statistic else "pcor"

if (method == "dynamic" && !is.null(args$ts_file)) {
  raw <- read.csv(args$ts_file, check.names = FALSE)
  tcol <- grep("^time$", tolower(names(raw)))
  if (length(tcol) == 0)
    stop("genenet_method='dynamic' needs a 'time' column in the time-series file.",
         call. = FALSE)
  ex <- as.matrix(raw[, -tcol[1], drop = FALSE])
  cat("GeneNet: dynamic mode,", nrow(ex), "time points\n")
  pcor_mat <- ggm.estimate.pcor(ex, method = "dynamic")
} else {
  ex <- ReadExpression(args$in_file, samples_as_rows = TRUE)
  cat("GeneNet: static mode,", nrow(ex), "observations,", ncol(ex), "genes\n")
  pcor_mat <- ggm.estimate.pcor(ex)
}

genes <- colnames(ex)

if (statistic == "prob") {
  # network.test.edges returns a long-format table already, with prob per edge.
  edges <- network.test.edges(pcor_mat, plot = FALSE, verbose = FALSE)
  conf <- data.frame(
    source = genes[edges$node1],
    target = genes[edges$node2],
    conf   = as.numeric(edges$prob),
    stringsAsFactors = FALSE
  )
  # GGMs are undirected; emit both orientations so the GCN sees a symmetric
  # input rather than silently losing half the edges.
  conf <- rbind(conf, data.frame(source = conf$target, target = conf$source,
                                 conf = conf$conf, stringsAsFactors = FALSE))
  conf <- conf[conf$source != conf$target & conf$conf > 0, ]
  conf <- conf[order(conf$conf, decreasing = TRUE), ]
  WriteList(conf, args$output_folder, "GENENET")
} else {
  mat <- abs(as.matrix(pcor_mat))
  dimnames(mat) <- list(genes, genes)
  conf <- expand.grid(source = genes, target = genes, stringsAsFactors = FALSE)
  conf$conf <- as.vector(mat)
  WriteList(ProcessList(conf), args$output_folder, "GENENET")
}
