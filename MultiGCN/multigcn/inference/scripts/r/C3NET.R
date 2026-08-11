# C3NET - conservative causal core network inference.
# Adapted from GENECI (MIT, (c) 2022 Adrian Segura Ortiz).

local({
  a <- commandArgs(trailingOnly = FALSE)
  self <- sub("^--file=", "", a[grep("^--file=", a)])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("c3net", "install.packages('c3net')")
suppressMessages(library(c3net))

# c3net expects genes as rows.
ex <- as.data.frame(ReadExpression(args$in_file, samples_as_rows = FALSE))
cat("C3NET:", nrow(ex), "genes,", ncol(ex), "samples\n")

call_args <- list(dataset = ex)
call_args <- AddIfSet(call_args, p, "alpha", cast = as.numeric)
call_args <- AddIfSet(call_args, p, "cutoffMI", cast = as.numeric)

network <- do.call(c3net, call_args)
WriteList(ProcessList(GetConfList(network)), args$output_folder, "C3NET")
