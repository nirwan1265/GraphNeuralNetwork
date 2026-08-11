# LEAP - lag-based expression association.
#
# Adapted from GENECI (MIT, (c) 2022 Adrian Segura Ortiz).
#
# LEAP correlates genes across a range of time lags and keeps the maximum
# absolute correlation (MAC). It assumes the columns are ordered in time - on
# unordered steady-state data it still runs, but the "lag" is meaningless and
# it becomes extremely permissive. Historically the densest output of the
# original 16, so expect the top-X% threshold to be doing a lot of work here.

local({
  a <- commandArgs(trailingOnly = FALSE)
  self <- sub("^--file=", "", a[grep("^--file=", a)])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("LEAP", "install.packages('LEAP')")
suppressMessages(library(LEAP))

# LEAP expects genes as rows, ordered columns.
ex <- as.data.frame(ReadExpression(args$in_file, samples_as_rows = FALSE))
genes <- rownames(ex)
cat("LEAP:", length(genes), "genes,", ncol(ex), "ordered columns\n")

rownames(ex) <- NULL
call_args <- list(data = ex)
call_args <- AddIfSet(call_args, p, "max_lag_prop", cast = as.numeric)
call_args <- AddIfSet(call_args, p, "MAC_cutoff", cast = as.numeric)

leap_table <- do.call(MAC_counter, call_args)

conf <- data.frame(
  source = genes[leap_table[, 3]],
  target = genes[leap_table[, 4]],
  conf   = abs(leap_table[, 1]),
  stringsAsFactors = FALSE
)

WriteList(ProcessList(conf), args$output_folder, "LEAP")
