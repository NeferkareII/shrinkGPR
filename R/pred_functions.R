#' Evaluate Predictive Densities
#'
#' \code{eval_pred_dens} evaluates the predictive density for a set of points based on a fitted \code{shrinkGPR} model.
#'
#' @param x Numeric vector of points for which the predictive density is to be evaluated.
#' @param mod A \code{shrinkGPR} object representing the fitted Gaussian process regression model.
#' @param data_test Data frame with one row containing the covariates for the test set.
#' Variables in \code{data_test} must match those used in model fitting.
#' @param nsamp Positive integer specifying the number of posterior samples to use for the evaluation. Default is 100.
#' @param log Logical; if \code{TRUE}, returns the log predictive density. Default is \code{FALSE}.
#' @return A numeric vector containing the predictive densities (or log predictive densities) for the points in \code{x}.
#' @details
#' This function computes predictive densities by marginalizing over posterior samples drawn from the fitted model. If the mean equation is included in the model, the corresponding covariates are incorporated.
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   # Simulate data
#'   set.seed(123)
#'   torch::torch_manual_seed(123)
#'   n <- 100
#'   x <- matrix(runif(n * 2), n, 2)
#'   y <- sin(2 * pi * x[, 1]) + rnorm(n, sd = 0.1)
#'   data <- data.frame(y = y, x1 = x[, 1], x2 = x[, 2])
#'
#'   # Fit GPR model
#'   res <- shrinkGPR(y ~ x1 + x2, data = data)
#'
#'   # Create point at which to evaluate predictive density
#'   data_test <- data.frame(x1 = 0.8, x2 = 0.5)
#'   eval_points <- c(-1.2, -1, 0)
#'
#'   eval_pred_dens(eval_points, res, data_test)
#'
#'   # Is vectorized, can also be used in functions like curve
#'   curve(eval_pred_dens(x, res, data_test), from = -1.5, to = -0.5)
#'   abline(v = sin(2 * pi * 0.8), col = "red")
#'   }
#' }
#' @export
eval_pred_dens <- function(x, mod, data_test, nsamp = 100, log = FALSE){

  # Input checking for eval_pred_dens --------------------------------------

  # Check that x is numeric
  if (!is.numeric(x)) {
    stop("The argument 'x' must be a numeric vector.")
  }

  # Check that mod is a shrinkGPR object
  if (!inherits(mod, "shrinkGPR")) {
    stop("The argument 'mod' must be an object of class 'shrinkGPR'.")
  }

  # Check that data_test is a data frame with one row
  if (!is.data.frame(data_test) || nrow(data_test) != 1) {
    stop("The argument 'data_test' must be a data frame with exactly one row.")
  }

  # Check that nsamp is a positive integer
  if (!is.numeric(nsamp) || nsamp <= 0 || nsamp %% 1 != 0) {
    stop("The argument 'nsamp' must be a positive integer.")
  }

  # Check that log is a logical value
  if (!is.logical(log) || length(log) != 1) {
    stop("The argument 'log' must be a single logical value.")
  }

  device <- attr(mod, "device")

  terms <- delete.response(mod$model_internals$terms)
  m <- model.frame(terms, data = data_test, xlev = mod$model_internals$xlevels)
  x_test <- torch_tensor(model.matrix(terms, m), device = device)

  if (mod$model_internals$x_mean) {
    terms_mean <- delete.response(mod$model_internals$terms_mean)
    m_mean <- model.frame(terms_mean, data = data_test, xlev = mod$model_internals$xlevels_mean)
    x_test_mean <- torch_tensor(model.matrix(terms_mean, m_mean), device = device)
  } else {
    x_test_mean <- NULL
  }

  x_tens <- torch_tensor(x, device = device)

  res_tens <- mod$model$eval_pred_dens(x_tens, x_test, nsamp, x_test_mean, log)
  return(as.numeric(res_tens))
}

#' Log Predictive Density Score
#'
#' \code{LPDS} calculates the log predictive density score for a fitted \code{shrinkGPR} model using a test dataset.
#'
#' @param mod A \code{shrinkGPR} object representing the fitted Gaussian process regression model.
#' @param data_test Data frame with one row containing the covariates for the test set.
#' Variables in \code{data_test} must match those used in model fitting.
#' @param nsamp Positive integer specifying the number of posterior samples to use for the evaluation. Default is 100.
#' @return A numeric value representing the log predictive density score for the test dataset.
#' @details
#' The log predictive density score is a measure of model fit that evaluates how well the model predicts unseen data.
#' It is computed as the log of the marginal predictive density of the observed responses.
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   # Simulate data
#'   set.seed(123)
#'   torch::torch_manual_seed(123)
#'   n <- 100
#'   x <- matrix(runif(n * 2), n, 2)
#'   y <- sin(2 * pi * x[, 1]) + rnorm(n, sd = 0.1)
#'   data <- data.frame(y = y, x1 = x[, 1], x2 = x[, 2])
#'
#'   # Fit GPR model
#'   res <- shrinkGPR(y ~ x1 + x2, data = data)
#'
#'   # Calculate true y value and calculate LPDS at specific point
#'   x1_new <- 0.8
#'   x2_new <- 0.5
#'   y_true <- sin(2 * pi * x1_new)
#'   data_test <- data.frame(y = y_true, x1 = x1_new, x2 = x2_new)
#'   LPDS(res, data_test)
#'   }
#' }
#' @export
LPDS <- function(mod, data_test, nsamp = 100) {

  # Input checking for LPDS -------------------------------------------------

  # Check that mod is a shrinkGPR object
  if (!inherits(mod, "shrinkGPR")) {
    stop("The argument 'mod' must be an object of class 'shrinkGPR'.")
  }

  # Check that data_test is a data frame with one row
  if (!is.data.frame(data_test) || nrow(data_test) != 1) {
    stop("The argument 'data_test' must be a data frame with exactly one row.")
  }

  # Check that nsamp is a positive integer
  if (!is.numeric(nsamp) || nsamp <= 0 || nsamp %% 1 != 0) {
    stop("The argument 'nsamp' must be a positive integer.")
  }

  # Create Vector y
  terms <- mod$model_internals$terms
  m <- model.frame(terms, data = data_test, xlev = mod$model_internals$xlevels)
  y <- model.response(m, "numeric")

  eval_pred_dens(y, mod, data_test, nsamp, log = TRUE)
}

#' Calculate Predictive Moments
#'
#' \code{calc_pred_moments} calculates the predictive means and variances for a fitted \code{shrinkGPR} model at new data points.
#'
#' @param object A \code{shrinkGPR} object representing the fitted Gaussian process regression model.
#' @param newdata \emph{Optional} data frame containing the covariates for the new data points. If missing, the training data is used.
#' @param nsamp Positive integer specifying the number of posterior samples to use for the calculation. Default is 100.
#' @return A list with two elements:
#' \itemize{
#'   \item \code{means}: A matrix of predictive means for each new data point, with the rows being the samples and the columns the data points.
#'   \item \code{vars}: An array of covariance matrices, with the first dimension corresponding to the samples and second and third dimensions to the data points.
#' }
#' @details
#' This function computes predictive moments by marginalizing over posterior samples from the fitted model. If the mean equation is included in the model, the corresponding covariates are used.
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   # Simulate data
#'   set.seed(123)
#'   torch::torch_manual_seed(123)
#'   n <- 100
#'   x <- matrix(runif(n * 2), n, 2)
#'   y <- sin(2 * pi * x[, 1]) + rnorm(n, sd = 0.1)
#'   data <- data.frame(y = y, x1 = x[, 1], x2 = x[, 2])
#'
#'   # Fit GPR model
#'   res <- shrinkGPR(y ~ x1 + x2, data = data)
#'
#'   # Calculate predictive moments
#'   momes <- calc_pred_moments(res, nsamp = 100)
#'   }
#' }
#' @export
calc_pred_moments <- function(object, newdata, nsamp = 100) {

  # Input checking for calc_pred_moments ------------------------------------

  # Check that object is a shrinkGPR object
  if (!inherits(object, "shrinkGPR")) {
    stop("The argument 'object' must be an object of class 'shrinkGPR'.")
  }

  # Check that newdata, if provided, is a data frame
  if (!missing(newdata) && !is.data.frame(newdata)) {
    stop("The argument 'newdata', if provided, must be a data frame.")
  }

  # Check that nsamp is a positive integer
  if (!is.numeric(nsamp) || nsamp <= 0 || nsamp %% 1 != 0) {
    stop("The argument 'nsamp' must be a positive integer.")
  }

  if (missing(newdata)) {
    newdata <- object$model_internals$data
  }

  device <- attr(object, "device")

  terms <- delete.response(object$model_internals$terms)
  m <- model.frame(terms, data = newdata, xlev = object$model_internals$xlevels)
  x_tens <- torch_tensor(model.matrix(terms, m), device = device)

  if (object$model_internals$x_mean) {
    terms_mean <- delete.response(object$model_internals$terms_mean)
    m_mean <- model.frame(terms_mean, data = newdata, xlev = object$model_internals$xlevels_mean)
    x_tens_mean <- torch_tensor(model.matrix(terms_mean, m_mean), device = device)
  } else {
    x_tens_mean <- NULL
  }

  res_tens <- object$model$calc_pred_moments(x_tens, nsamp, x_tens_mean)

  return(list(means = as.matrix(res_tens[[1]]),
              vars = as.array(res_tens[[2]])))
}

#' Generate Predictions
#'
#' \code{predict.shrinkGPR} generates posterior predictive samples from a fitted \code{shrinkGPR} model at specified covariates.
#'
#' @param object A \code{shrinkGPR} object representing the fitted Gaussian process regression model.
#' @param newdata \emph{Optional} data frame containing the covariates for the prediction points. If missing, the training data is used.
#' @param nsamp Positive integer specifying the number of posterior samples to generate. Default is 100.
#' @param ... Currently ignored.
#' @return A matrix containing posterior predictive samples for each covariate combination in \code{newdata}.
#' @details
#' This function generates predictions by sampling from the posterior predictive distribution. If the mean equation is included in the model, the corresponding covariates are incorporated.
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   # Simulate data
#'   set.seed(123)
#'   torch::torch_manual_seed(123)
#'   n <- 100
#'   x <- matrix(runif(n * 2), n, 2)
#'   y <- sin(2 * pi * x[, 1]) + rnorm(n, sd = 0.1)
#'   data <- data.frame(y = y, x1 = x[, 1], x2 = x[, 2])
#'
#'   # Fit GPR model
#'   res <- shrinkGPR(y ~ x1 + x2, data = data)
#'   # Example usage for in-sample prediction
#'   preds <- predict(res)
#'
#'   # Example usage for out-of-sample prediction
#'   newdata <- data.frame(x1 = runif(10), x2 = runif(10))
#'   preds <- predict(res, newdata = newdata)
#'   }
#' }
#' @export
predict.shrinkGPR <- function(object, newdata, nsamp = 100, ...) {

  # Input checking for predict.shrinkGPR ------------------------------------

  # Check that object is a shrinkGPR object
  if (!inherits(object, "shrinkGPR")) {
    stop("The argument 'object' must be an object of class 'shrinkGPR'.")
  }

  # Check that newdata, if provided, is a data frame
  if (!missing(newdata) && !is.data.frame(newdata)) {
    stop("The argument 'newdata', if provided, must be a data frame.")
  }

  # Check that nsamp is a positive integer
  if (!is.numeric(nsamp) || nsamp <= 0 || nsamp %% 1 != 0) {
    stop("The argument 'nsamp' must be a positive integer.")
  }

  if (missing(newdata)) {
    newdata <- object$model_internals$data
  }

  device <- attr(object, "device")

  terms <- delete.response(object$model_internals$terms)
  m <- model.frame(terms, data = newdata, xlev = object$model_internals$xlevels)
  x_tens <- torch_tensor(model.matrix(terms, m), device = device)

  if (object$model_internals$x_mean) {
    terms_mean <- delete.response(object$model_internals$terms_mean)
    m_mean <- model.frame(terms_mean, data = newdata, xlev = object$model_internals$xlevels_mean)
    x_tens_mean <- torch_tensor(model.matrix(terms_mean, m_mean), device = device)
  } else {
    x_tens_mean <- NULL
  }

  res_tens <- object$model$predict(x_tens, nsamp, x_tens_mean)

  return(as.matrix(res_tens))
}

#' Generate Posterior Samples
#'
#' \code{gen_posterior_samples} generates posterior samples of the model parameters from a fitted \code{shrinkGPR} model.
#'
#' @param mod A \code{shrinkGPR} object representing the fitted Gaussian process regression model.
#' @param nsamp Positive integer specifying the number of posterior samples to generate. Default is 1000.
#' @return A list containing posterior samples of the model parameters:
#' \itemize{
#'   \item \code{thetas}: A matrix of posterior samples for the inverse lengthscale parameters.
#'   \item \code{sigma2}: A matrix of posterior samples for the noise variance.
#'   \item \code{lambda}: A matrix of posterior samples for the global shrinkage parameter.
#'   \item \code{betas} (optional): A matrix of posterior samples for the mean equation parameters (if included in the model).
#'   \item \code{lambda_mean} (optional): A matrix of posterior samples for the mean equation's global shrinkage parameter (if included in the model).
#' }
#' @details
#' This function draws posterior samples from the latent space and transforms them into the parameter space of the model. These samples can be used for posterior inference or further analysis.
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   # Simulate data
#'   set.seed(123)
#'   torch::torch_manual_seed(123)
#'   n <- 100
#'   x <- matrix(runif(n * 2), n, 2)
#'   y <- sin(2 * pi * x[, 1]) + rnorm(n, sd = 0.1)
#'   data <- data.frame(y = y, x1 = x[, 1], x2 = x[, 2])
#'
#'   # Fit GPR model
#'   res <- shrinkGPR(y ~ x1 + x2, data = data)
#'
#'   # Generate posterior samples
#'   samps <- gen_posterior_samples(res, nsamp = 1000)
#'
#'   # Plot the posterior samples
#'   boxplot(samps$thetas)
#'   }
#' }
#' @export
gen_posterior_samples <- function(mod, nsamp = 1000) {

  # Input checking for gen_posterior_samples -------------------------------

  # Check that mod is a shrinkGPR object
  if (!inherits(mod, "shrinkGPR")) {
    stop("The argument 'mod' must be an object of class 'shrinkGPR'.")
  }

  # Check that nsamp is a positive integer
  if (!is.numeric(nsamp) || nsamp <= 0 || nsamp %% 1 != 0) {
    stop("The argument 'nsamp' must be a positive integer.")
  }

  z <- mod$model$gen_batch(nsamp)
  zk <- mod$model(z)[[1]]

  # Split into list containing groups of parameters
  # Convention:
  # First d_cov components are the theta parameters
  # Next component is the sigma parameter
  # Next component is the lambda parameter
  # Next d_mean components are the mean parameters
  # Last component is the lambda parameter for the mean

  d_cov <- mod$model_internals$d_cov

  res <- list(thetas = as.matrix(zk[, 1:d_cov]),
              sigma2 = as.matrix(zk[, d_cov + 1]),
              lambda = as.matrix(zk[, d_cov + 2]))


  if (mod$model_internals$x_mean) {
    d_mean <- mod$model_internals$d_mean
    res$betas <- as.matrix(zk[, (d_cov + 3):(d_cov + 2 + d_mean)])
    res$lambda_mean <- as.matrix(zk[, -1])
  }

  return(res)


}
