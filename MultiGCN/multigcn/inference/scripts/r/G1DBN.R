# MultiGCN wrapper for G1DBN (first-order dynamic Bayesian network).
#
# TIME SERIES REQUIRED. G1DBN regresses each gene's value at time t+1 on all
# genes at time t, so without ordered time points there is nothing to fit.
#
# Two-step procedure (Lebre 2009):
#   step 1  DBNScoreStep1 - score every pair with a first-order DBN
#   step 2  DBNScoreStep2 - refit only on pairs surviving alpha1
# alpha1 is deliberately loose: it is a screening threshold, not a decision.
# Final score is 1 - the step-2 p-value, so larger means more confident.

local({
  self <- sub("^--file=", "",
              commandArgs(trailingOnly = FALSE)[grep("^--file=",
                          commandArgs(trailingOnly = FALSE))])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("G1DBN", "install.packages('G1DBN')")
suppressMessages(library(G1DBN))

if (is.null(args$ts_file))
  stop("G1DBN requires time-series data. Pass time_series='...csv'.", call. = FALSE)

alpha1 <- if (!is.null(p$alpha1)) as.numeric(p$alpha1) else 0.5

raw <- read.csv(args$ts_file, check.names = FALSE)
tcol <- grep("^time$", tolower(names(raw)))
if (length(tcol) == 0)
  stop(paste("Expected a 'time' column in", args$ts_file), call. = FALSE)

# G1DBN expects a matrix with time points as rows and genes as columns.
ex <- as.matrix(raw[, -tcol[1], drop = FALSE])
genes <- colnames(ex)
cat("G1DBN:", nrow(ex), "time points,", length(genes), "genes\n")

if (nrow(ex) < 4)
  stop(sprintf("Only %d time points; G1DBN needs more to fit a DBN.", nrow(ex)),
       call. = FALSE)

S1 <- DBNScoreStep1(ex, method = "ls")
S2 <- DBNScoreStep2(S1$S1ls, data = ex, method = "ls", alpha1 = alpha1)

# S2 holds p-values; NA means the pair was screened out at step 1.
score <- 1 - S2
score[is.na(score)] <- 0
dimnames(score) <- list(genes, genes)

conf <- expand.grid(source = genes, target = genes, stringsAsFactors = FALSE)
conf$conf <- as.vector(score)
WriteList(ProcessList(conf), args$output_folder, "G1DBN")
