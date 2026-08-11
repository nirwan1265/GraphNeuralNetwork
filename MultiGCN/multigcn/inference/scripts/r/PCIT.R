# PCIT - partial correlation and information theory.
#
# Adapted from GENECI (MIT, (c) 2022 Adrian Segura Ortiz).
#
# Uses CeTF::PCIT (there is no package called "pcit"). Takes the absolute
# correlation: PCIT reports direction of correlation, but this pipeline ranks
# regulatory confidence, and a strong negative correlation is evidence of
# regulation just as much as a strong positive one.

local({
  a <- commandArgs(trailingOnly = FALSE)
  self <- sub("^--file=", "", a[grep("^--file=", a)])
  source(file.path(dirname(self), "_multigcn_common.R"))
}, envir = globalenv())

args <- ParseArgs()
p <- args$params
RequirePackage("CeTF", "BiocManager::install('CeTF')")
suppressMessages(library(CeTF))

tolType <- if (!is.null(p$tolType)) p$tolType else "mean"

# CeTF::PCIT expects genes as rows.
ex <- as.data.frame(ReadExpression(args$in_file, samples_as_rows = FALSE))
cat("PCIT:", nrow(ex), "genes,", ncol(ex), "samples, tolType =", tolType, "\n")

res <- PCIT(ex, tolType = tolType)

# tab is (gene1, gene2, corr, pcit_corr); drop the raw correlation column and
# keep the PCIT-significant one.
conf <- res$tab[, -3]
conf[, 3] <- abs(conf[, 3])

WriteList(ProcessList(conf), args$output_folder, "PCIT")
