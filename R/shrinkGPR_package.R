## usethis namespace: start
#'
#' @import torch
#'
#' @importFrom gsl hyperg_U
#'
#' @importFrom progress progress_bar
#'
#' @importFrom stats model.response model.matrix model.frame rnorm na.pass delete.response .getXlevels pt
#'
#' @importFrom methods formalArgs
#'
#' @importFrom utils packageVersion
#'
#'
## usethis namespace: end

.onAttach <- function(libname, pkgname) {

  if (!torch::torch_is_installed()) {
    packageStartupMessage("Welcome to shrinkGPR version ", packageVersion("shrinkGPR"),
                          ".\n \nNOTE: No torch installation detected. This package requires torch to function.",
                          "Please install torch by running torch::install_torch()")
  } else {
    if (cuda_is_available()) {
      CUDA_message <- "CUDA installation detected and functioning with torch.
CUDA will be used for GPU acceleration by default."
    } else {
      CUDA_message <- "NOTE: No CUDA installation detected. This may be quite slow for larger datasets.
Consider installing CUDA for GPU acceleration. Information on this can be found at:
https://cran.r-project.org/web/packages/torch/vignettes/installation.html"
    }

    start_message <- paste0("\nWelcome to shrinkGPR version ", packageVersion("shrinkGPR"),
                            ".\n \n", CUDA_message)

    packageStartupMessage(start_message)
  }


}
