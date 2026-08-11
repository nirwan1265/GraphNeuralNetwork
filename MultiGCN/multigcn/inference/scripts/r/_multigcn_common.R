# Shared helpers for MultiGCN-owned R wrappers.
# Written for MultiGCN (MIT) - not vendored from GENECI.

suppressMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE))
    install.packages("jsonlite", repos = "http://cran.us.r-project.org")
  library(jsonlite)
})

# Parse: <in_file> <output_folder> <json_params> [<time_series_file>]
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

# Rescale to [0,1] and drop zero-confidence rows, matching the convention every
# other algorithm in the pipeline follows.
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
