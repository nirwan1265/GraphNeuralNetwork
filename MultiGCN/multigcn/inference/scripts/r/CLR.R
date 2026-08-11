# CLR via Bioconductor minet. See _minet_core.R for the shared implementation.
local({
  a <- commandArgs(trailingOnly = FALSE)
  self <- sub("^--file=", "", a[grep("^--file=", a)])
  source(file.path(dirname(self), "_multigcn_common.R"))
  source(file.path(dirname(self), "_minet_core.R"))
}, envir = globalenv())

RunMinet("clr", "CLR")
