# Shared helpers for MultiGCN's R wrappers.
#
# Adapted from GENECI's components/infer_network/functions.R
# (https://github.com/AdrianSeguraOrtiz/GENECI), Copyright (c) 2022 Adrian
# Segura Ortiz, MIT License. See THIRD_PARTY_NOTICES.md.
#
# Changes from the original:
#   - GetConfList/ProcessList kept behaviourally identical so results match
#     the previously validated DREAM4 runs.
#   - added JSON parameter parsing, so algorithms are tunable rather than
#     hardcoded.
#   - dropped the gdata dependency, which was loaded but never used.
#   - output path no longer prefixed with "./", which broke absolute paths.

suppressMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE))
    install.packages("jsonlite", repos = "https://cloud.r-project.org")
  library(jsonlite)
})

# Parse: <in_file> <output_folder> [<json_params>] [<time_series_file>]
ParseArgs <- function() {
  a <- commandArgs(trailingOnly = TRUE)
  if (length(a) < 2)
    stop("Usage: Rscript <script>.R input.csv output_folder [params.json] [ts.csv]",
         call. = FALSE)
  list(
    in_file = a[1],
    output_folder = a[2],
    params = if (length(a) >= 3 && nzchar(a[3])) fromJSON(a[3]) else list(),
    ts_file = if (length(a) >= 4 && nzchar(a[4])) a[4] else NULL
  )
}

# Canonical input is GENES x SAMPLES. Most modelling packages want the
# transpose (samples as rows), so read once and be explicit about it.
ReadExpression <- function(path, samples_as_rows = TRUE) {
  m <- read.table(path, sep = ",", head = TRUE, row.names = 1,
                  check.names = FALSE)
  if (samples_as_rows) t(as.matrix(m)) else as.matrix(m)
}

# Square matrix -> long (source, target, confidence), sorted descending.
GetConfList <- function(network) {
  df <- data.frame(as.table(as.matrix(network)))
  df[order(df$Freq, decreasing = TRUE), ]
}

# Rescale to [0,1] and drop zero-confidence rows. Matches the convention every
# algorithm in the pipeline follows, so scores stay comparable across methods.
ProcessList <- function(df) {
  colnames(df) <- c("source", "target", "conf")
  df <- df[!is.na(df$conf), ]
  df <- df[df$source != df$target, ]
  rng <- range(df$conf)
  df$conf <- if (diff(rng) < 1e-12) 0 else (df$conf - rng[1]) / diff(rng)
  df <- df[df$conf != 0, ]
  df[order(df$conf, decreasing = TRUE), ]
}

WriteList <- function(df, output_folder, algorithm) {
  dir.create(output_folder, showWarnings = FALSE, recursive = TRUE)
  path <- file.path(output_folder, paste0("GRN_", algorithm, ".csv"))
  write.table(df, path, sep = ",", col.names = FALSE, row.names = FALSE,
              quote = FALSE)
  cat("wrote", nrow(df), "edges to", path, "\n")
}

RequirePackage <- function(pkg, how) {
  if (!requireNamespace(pkg, quietly = TRUE))
    stop(sprintf("Package '%s' is not installed. Install it with:\n  %s", pkg, how),
         call. = FALSE)
}

# Only forward a parameter to the underlying package when the user actually set
# it. Keeping unset parameters absent means every algorithm runs on its own
# library defaults - identical to the previously validated GENECI behaviour -
# and a wrong argument name can only surface for someone who opted in.
AddIfSet <- function(arglist, params, name, target = name, cast = identity) {
  if (!is.null(params[[name]])) arglist[[target]] <- cast(params[[name]])
  arglist
}

# Resolve a sibling script regardless of the working directory Rscript was
# invoked from.
SourceSibling <- function(filename) {
  a <- commandArgs(trailingOnly = FALSE)
  self <- sub("^--file=", "", a[grep("^--file=", a)])
  source(file.path(dirname(self), filename))
}
