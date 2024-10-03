#' @export
simGPR <-  function(N = 200, d = 3, d_mean = 3, sigma2 = 1, lam = 2, kernel_func = kernel_se, perc_spars = 0.5, theta, beta, device){


  # Add device attribute, set to GPU if available
  if (missing(device)) {
    if (cuda_is_available()) {
      device <- torch_device("cuda")
    } else {
      device <- torch_device("cpu")
    }
  }

  # Take user supplied beta/theta or generate them
  if (missing(beta)){
    beta <- rnorm(d_mean, 0, 1)

    # Sparsify
    beta[sample(1:d, round(d * perc_spars))] <- 0
  } else {
    if (is.vector(beta) == FALSE){
      stop("beta has to be a vector")
    }

    if (length(beta) != d){
      stop("beta has to be of length d")
    }
    # if (any(sapply(beta, numeric_input_bad_))){
    #   stop("all elements of beta have to be single numbers")
    # }
  }

  if (missing(theta)){
    theta <- rnorm(d, mean = 1.5)^2

    # Sparsify
    theta[sample(1:d, round(d * perc_spars))] <- 0
  } else {

    if (is.vector(theta) == FALSE){
      stop("theta has to be a vector")
    }

    if (length(theta) != d){
      stop("theta has to be of length d")
    }
    # if (any(sapply(theta, numeric_input_bad_zer))){
    #   stop("all elements of theta have to be scalars that are >= 0")
    # }
  }

  # Transform into torch tensors
  beta_tens <- torch_tensor(beta, device = device)$unsqueeze(2)
  theta_tens <- torch_tensor(theta, device = device)$unsqueeze(1)
  lam_tensor <- torch_tensor(lam, device = device)$unsqueeze(2)

  # Generate y
  x_tens <- torch_randn(N, d, device = device)
  K <- kernel_func(theta_tens, lam_tensor, x_tens)$squeeze() + sigma2 * torch_eye(N, device = device)

  if (d_mean > 0) {
    x_mean <- torch_randn(N, d_mean, device = device)
    mu <- torch_mm(x_mean, beta_tens)
  } else {
    mu <- torch_zeros(N, 1, device = device)
  }


  multivar_dist <- distr_multivariate_normal(mu$squeeze(), K)

  y <- multivar_dist$sample()

  if (d_mean > 0) {
    data_frame <- data.frame(y = as.numeric(y),
                             x = as.matrix(x_tens),
                             x_mean = as.matrix(x_mean))
  } else {
    data_frame <- data.frame(y = as.numeric(y),
                             x = as.matrix(x_tens))
  }

  res <- list(data = data_frame,
              true_vals = list(theta = theta,
                               sigma2 = sigma2,
                               lam = lam))

  if (d_mean > 0) {
    res$true_vals$beta <- beta
  }

  return(res)
}
