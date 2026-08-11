# MultiGCN wrapper for ppcor (partial correlation).
#
# Partial correlation measures the association between two genes with every
# other gene held constant, so it removes the indirect edges that plain
# correlation reports (A->B->C showing up as a spurious A-C link).
#
# Caveat: ppcor inverts the covariance matrix, which needs more observations
# than genes. On wide data (genes >> samples) it is singular or unstable. Use
# GENENET instead in that regime - it shrinks the estimate and stays defined.

local({
  self <- sub("^--file=", "",
              commandArgs(trailingOnly = FALSE)[grep("^--file=",
                          commandArgs(trailingOnly = FALSE))])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("ppcor", "install.packages('ppcor')")
suppressMessages(library(ppcor))

method    <- if (!is.null(p$method)) p$method else "pearson"
statistic <- if (!is.null(p$statistic)) p$statistic else "estimate"

# ppcor wants observations as rows, variables (genes) as columns.
ex <- as.data.frame(ReadExpression(args$in_file, samples_as_rows = TRUE))
n_obs <- nrow(ex); n_genes <- ncol(ex)
cat("ppcor:", n_obs, "observations,", n_genes, "genes\n")

if (n_obs <= n_genes)
  warning(sprintf(
    paste("ppcor needs more observations (%d) than genes (%d) to invert the",
          "covariance matrix. Result may be singular or unstable; consider",
          "GENENET instead."), n_obs, n_genes))

res <- tryCatch(pcor(ex, method = method), error = function(e)
  stop(paste("ppcor failed:", conditionMessage(e),
             "\nThis is usually a singular covariance matrix. Use GENENET."),
       call. = FALSE))

mat <- if (statistic == "p.value") 1 - res$p.value else abs(res$estimate)
genes <- colnames(ex)
dimnames(mat) <- list(genes, genes)

conf <- expand.grid(source = genes, target = genes, stringsAsFactors = FALSE)
conf$conf <- as.vector(mat)
WriteList(ProcessList(conf), args$output_folder, "PPCOR")
