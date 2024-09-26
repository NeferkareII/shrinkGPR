#' @export
eval_pred_dens <- function(x, mod, data_test, nsamp = 100, log = FALSE){

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

#' @export
LPDS <- function(mod, data_test, nsamp = 100) {

  # Create Vector y
  terms <- mod$model_internals$terms
  m <- model.frame(terms, data = data_test, xlev = mod$model_internals$xlevels)
  y <- model.response(m, "numeric")

  eval_pred_dens(y, mod, data_test, nsamp, log = TRUE)
}

#' @export
calc_pred_moments <- function(object, newdata, nsamp) {

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
              vars = as.matrix(res_tens[[2]])))
}

#' @export
predict.shrinkGPR <- function(object, newdata, nsamp = 100) {

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

#' @export
gen_posterior_samples <- function(mod, nsamp = 1000) {
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
