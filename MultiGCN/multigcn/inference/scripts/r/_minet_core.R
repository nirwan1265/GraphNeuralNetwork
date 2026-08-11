# Shared core for the four minet-based algorithms: ARACNE, CLR, MRNET, MRNETB.
#
# All four are the same call with a different `method` string, so they live in
# one place rather than four near-identical copies (which is how GENECI ships
# them, and how they drift apart).
#
# Adapted from GENECI (MIT, (c) 2022 Adrian Segura Ortiz).
# Behaviour at default parameters is identical to the original.

RunMinet <- function(method, algorithm) {
  args <- ParseArgs()
  p <- args$params

  RequirePackage("minet", "BiocManager::install('minet')")
  suppressMessages(library(minet))

  # minet expects samples as rows, genes as columns.
  ex <- ReadExpression(args$in_file, samples_as_rows = TRUE)
  cat(algorithm, ":", nrow(ex), "samples x", ncol(ex), "genes\n")

  call_args <- list(dataset = ex, method = method)
  call_args <- AddIfSet(call_args, p, "estimator")
  call_args <- AddIfSet(call_args, p, "disc")
  call_args <- AddIfSet(call_args, p, "nbins", cast = as.integer)

  network <- do.call(minet, call_args)
  WriteList(ProcessList(GetConfList(network)), args$output_folder, algorithm)
}
