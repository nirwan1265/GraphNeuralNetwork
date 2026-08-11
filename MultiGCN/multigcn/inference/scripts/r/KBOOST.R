# KBOOST - kernel PCA and boosting.
# Adapted from GENECI (MIT, (c) 2022 Adrian Segura Ortiz).

local({
  a <- commandArgs(trailingOnly = FALSE)
  self <- sub("^--file=", "", a[grep("^--file=", a)])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("KBoost", "BiocManager::install('KBoost')")
suppressMessages(library(KBoost))

# KBoost expects samples as rows, genes as columns.
ex <- ReadExpression(args$in_file, samples_as_rows = TRUE)
cat("KBOOST:", nrow(ex), "samples x", ncol(ex), "genes\n")

call_args <- list(X = ex)
call_args <- AddIfSet(call_args, p, "g", cast = as.numeric)
call_args <- AddIfSet(call_args, p, "v", cast = as.numeric)
call_args <- AddIfSet(call_args, p, "ite", cast = as.integer)

network <- do.call(kboost, call_args)[["GRN"]]
WriteList(ProcessList(GetConfList(network)), args$output_folder, "KBOOST")
