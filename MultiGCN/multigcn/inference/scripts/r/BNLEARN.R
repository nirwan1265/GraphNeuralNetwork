# MultiGCN wrapper for bnlearn Bayesian network structure learning.
#
# Why boot.strength() rather than a single hc()/tabu() fit:
# a single structure fit returns one DAG, i.e. binary edges. This pipeline
# ranks gene pairs by continuous confidence and takes the top X%, so a binary
# input carries almost no ranking information - every retained edge would tie.
# Bootstrap model averaging (Friedman et al., 1999) refits the structure over R
# resamples and reports how often each edge appears, giving a genuine [0,1]
# confidence. Score = strength * direction, so the result stays directed.

local({
  self <- sub("^--file=", "",
              commandArgs(trailingOnly = FALSE)[grep("^--file=",
                          commandArgs(trailingOnly = FALSE))])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("bnlearn", "install.packages('bnlearn')")
suppressMessages(library(bnlearn))

algorithm <- if (!is.null(p$algorithm)) p$algorithm else "hc"
R_boot    <- if (!is.null(p$R)) as.integer(p$R) else 200
score     <- if (!is.null(p$score)) p$score else "bic-g"
restart   <- if (!is.null(p$restart)) as.integer(p$restart) else 0

# cluster = 0 means "auto": use every core but one, so the machine stays
# usable during what is often an hour-plus run. Hardcoding a number would be
# wrong on any machine with a different core count.
n_cluster <- if (!is.null(p$cluster)) as.integer(p$cluster) else 0L
if (n_cluster == 0L) {
  detected <- tryCatch(parallel::detectCores(logical = FALSE),
                       error = function(e) NA_integer_)
  if (is.na(detected) || detected < 2L) detected <- 1L
  n_cluster <- max(1L, detected - 1L)
  cat("bnlearn: auto-detected", detected, "physical cores, using", n_cluster, "\n")
}

# bnlearn expects a data.frame with variables (genes) as COLUMNS.
ex <- as.data.frame(ReadExpression(args$in_file, samples_as_rows = TRUE))

if (nrow(ex) < 10)
  warning(sprintf(
    "Only %d observations for %d genes. Bayesian structure learning is unreliable at this ratio.",
    nrow(ex), ncol(ex)))

# Score-based methods take score/restart; constraint-based ones do not.
algo_args <- if (algorithm %in% c("hc", "tabu")) {
  list(score = score, restart = restart)
} else list()

cl <- NULL
if (n_cluster > 1) {
  RequirePackage("parallel", "install.packages('parallel')")
  cl <- parallel::makeCluster(n_cluster)
  on.exit(parallel::stopCluster(cl), add = TRUE)
}

cat("bnlearn: boot.strength with", algorithm, "R =", R_boot,
    "on", n_cluster, "worker(s)\n")
cat("  this is", R_boot, "complete structure searches - expect it to be slow\n")
start_time <- Sys.time()

str_df <- bnlearn::boot.strength(
  data = ex, R = R_boot, algorithm = algorithm,
  algorithm.args = algo_args, cluster = cl
)

cat("  finished in",
    format(round(difftime(Sys.time(), start_time, units = "mins"), 1)), "\n")

# strength = P(edge present); direction = P(oriented this way | present).
conf <- data.frame(
  source = as.character(str_df$from),
  target = as.character(str_df$to),
  conf   = as.numeric(str_df$strength) * as.numeric(str_df$direction),
  stringsAsFactors = FALSE
)

WriteList(ProcessList(conf), args$output_folder, "BNLEARN")
