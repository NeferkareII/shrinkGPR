#' @title Kernel Functions for Gaussian Processes
#' @description
#' A set of kernel functions for Gaussian processes, including the squared exponential (SE) kernel and Matérn kernels
#' with smoothness parameters 1/2, 3/2, and 5/2. These kernels compute the covariance structure for Gaussian process regression
#' models and are designed for compatibility with the \code{shrinkGPR} function.
#' @name kernel_functions
#' @param thetas A \code{torch_tensor} of dimensions \code{n_latent x d}, representing the latent length-scale parameters.
#' @param tau A \code{torch_tensor} of length \code{n_latent}, representing the latent scaling factors.
#' @param x A \code{torch_tensor} of dimensions \code{N x d}, containing the input data points.
#' @param x_star Either \code{NULL} or a \code{torch_tensor} of dimensions \code{N_new x d}. If \code{NULL}, the kernel is computed
#' for \code{x} against itself. Otherwise, it computes the kernel between \code{x} and \code{x_star}.
#' @return A \code{torch_tensor} containing the batched covariance matrices (one for each latent sample):
#' \itemize{
#'   \item If \code{x_star = NULL}, the output is of dimensions \code{n_latent x N x N}, representing pairwise covariances between all points in \code{x}.
#'   \item If \code{x_star} is provided, the output is of dimensions \code{n_latent x N_new x N}, representing pairwise covariances between \code{x_star} and \code{x}.
#' }
#' @details
#' These kernel functions are used to define the covariance structure in Gaussian process regression models. Each kernel implements a specific covariance function:
#' \itemize{
#'   \item \code{kernel_se}: Squared exponential (SE) kernel, also known as the radial basis function (RBF) kernel.
#'   It assumes smooth underlying functions.
#'   \item \code{kernel_matern_12}: Matérn kernel with smoothness parameter \eqn{\nu = 1/2}, equivalent to the absolute exponential kernel.
#'   \item \code{kernel_matern_32}: Matérn kernel with smoothness parameter \eqn{\nu = 3/2}.
#'   \item \code{kernel_matern_52}: Matérn kernel with smoothness parameter \eqn{\nu = 5/2}.
#' }
#'
#' The \code{sqdist} helper function is used internally by these kernels to compute squared distances between data points.
#'
#' Note that these functions perform no input checks, as to ensure higher performance.
#' Users should ensure that the input tensors are of the correct dimensions.
#' @examples
#' # Example inputs
#' torch_manual_seed(123)
#' n_latent <- 3
#' d <- 2
#' N <- 5
#' thetas <- torch_randn(n_latent, d)$abs()
#' tau <- torch_randn(n_latent)$abs()
#' x <- torch_randn(N, d)
#'
#' # Compute the SE kernel
#' K_se <- kernel_se(thetas, tau, x)
#' print(K_se)
#'
#' # Compute the Matérn 3/2 kernel
#' K_matern32 <- kernel_matern_32(thetas, tau, x)
#' print(K_matern32)
#'
#' # Compute the Matérn 5/2 kernel with x_star
#' x_star <- torch_randn(3, d)
#' K_matern52 <- kernel_matern_52(thetas, tau, x, x_star)
#' print(K_matern52)
NULL

sqdist <- function(x, thetas, x_star = NULL) {
  X_thetas <- x$unsqueeze(3) * torch_sqrt(thetas$t())
  sq <- torch_sum(X_thetas^2, 2, keepdim = TRUE)

  if (is.null(x_star)) {

    sqdist <- (sq + sq$permute(c(2, 1, 3)))$permute(c(3, 1, 2)) -
      2 * torch_bmm(X_thetas$permute(c(3, 1, 2)), X_thetas$permute(c(3, 2, 1)))

  } else {

    X_star_thetas <- x_star$unsqueeze(3) * torch_sqrt(thetas$t())
    sq_star <- torch_sum(X_star_thetas^2, 2, keepdim = TRUE)
    sqdist <- (sq_star + sq$permute(c(2, 1, 3)))$permute(c(3, 1, 2)) -
      2 * torch_bmm(X_star_thetas$permute(c(3, 1, 2)), X_thetas$permute(c(3, 2, 1)))

  }
  return(sqdist)
}

#' @rdname kernel_functions
#' @export
kernel_se <- function(thetas, tau, x, x_star = NULL) {
  sqdist_x <- sqdist(x, thetas, x_star)
  K <- 1/(tau$unsqueeze(2)$unsqueeze(2)) * torch_exp(-0.5 * sqdist_x)

  return(K)
}

#' @rdname kernel_functions
#' @export
kernel_matern_12 <- function(thetas, tau, x, x_star = NULL) {
  sqdist <- torch_sqrt(sqdist(x, thetas, x_star) + 1e-4)
  K <- 1/(tau$unsqueeze(2)$unsqueeze(2)) * torch_exp(-sqdist)

  return(K)
}

#' @rdname kernel_functions
#' @export
kernel_matern_32 <- function(thetas, tau, x, x_star = NULL) {
  sqdist <- torch_sqrt(sqdist(x, thetas, x_star) + 1e-4)
  K <- 1/(tau$unsqueeze(2)$unsqueeze(2)) * (1 + sqrt(3) * sqdist) * torch_exp(-sqrt(3) * sqdist)

  return(K)
}

#' @rdname kernel_functions
#' @export
kernel_matern_52 <- function(thetas, tau, x, x_star = NULL) {
  sqdist <- torch_sqrt(sqdist(x, thetas, x_star) + 1e-4)
  K <- 1/(tau$unsqueeze(2)$unsqueeze(2)) * (1 + sqrt(5) * sqdist + 5/3 * sqdist^2) * torch_exp(-sqrt(5) * sqdist)

  return(K)
}
