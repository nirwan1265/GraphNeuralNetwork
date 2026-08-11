# BC3NET - bootstrap C3NET.
#
# Adapted from GENECI (MIT, (c) 2022 Adrian Segura Ortiz).
#
# Note on the replicate loop: bc3net already bootstraps internally, and GENECI
# then runs the whole thing 11 more times and takes the per-edge median. That
# is redundant in principle but it is what produced the validated DREAM4
# numbers, so it stays the default. Set bc3net_replicates=1 to skip it - much
# faster, and worth checking whether it actually changes anything.

local({
  a <- commandArgs(trailingOnly = FALSE)
  self <- sub("^--file=", "", a[grep("^--file=", a)])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("bc3net", "install.packages('bc3net')")
suppressMessages(library(bc3net))

replicates <- if (!is.null(p$replicates)) as.integer(p$replicates) else 11L

# bc3net expects genes as rows (it transposes internally).
ex <- as.data.frame(ReadExpression(args$in_file, samples_as_rows = FALSE))
cat("BC3NET:", nrow(ex), "genes,", ncol(ex), "samples,", replicates, "replicate(s)\n")

run_once <- function() {
  call_args <- list(dataset = ex, igraph = FALSE)
  call_args <- AddIfSet(call_args, p, "boot", cast = as.integer)
  call_args <- AddIfSet(call_args, p, "estimator")
  GetConfList(do.call(bc3net, call_args))
}

dt <- run_once()
colnames(dt)[3] <- "conf_1"

if (replicates > 1) {
  for (i in 2:replicates) {
    cat("  replicate", i, "of", replicates, "\n")
    dt <- merge(dt, run_once(), by = c(1, 2), all.x = TRUE, all.y = TRUE,
                suffixes = c("", paste0("_", i)))
  }
  conf <- data.frame(dt[, 1], dt[, 2],
                     apply(dt[, -c(1, 2), drop = FALSE], 1, median, na.rm = TRUE))
} else {
  conf <- dt
}

WriteList(ProcessList(conf), args$output_folder, "BC3NET")
