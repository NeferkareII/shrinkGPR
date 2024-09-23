# Create nn_module subclass that implements forward methods for GPR
#' @export
GPR_class <- nn_module(
  classname = "GPR",
  initialize = function(y, x, x_mean,
                        a = 0.5, c = 0.5, a_mean = 0.5, c_mean = 0.5, sigma2_rate = 10,
                        n_layers, flow_func, flow_args,
                        kernel_func = kernel_se, device) {

    # Add dimension attribute
    # size of x +1 for the variance term + 1 for global shrinkage parameter
    # + size of x_mean, if provided

    self$d <- ncol(x) + 2
    self$mean_zero <- TRUE
    if (!missing(x_mean)) {
      self$d <- self$d + ncol(x_mean)
      self$mean_zero <- FALSE
    }

    self$N <- nrow(x)

    # Add kernel attribute
    self$kernel_func <- kernel_func

    # Add device attribute, set to GPU if available
    if (missing(device)) {
      if (cuda_is_available()) {
        self$device <- torch_device("cuda")
      } else {
        self$device <- torch_device("cpu")
      }
    } else {
      self$device <- device
    }

    # Add softplus function for positive parameters
    self$beta_sp <- 0.7
    self$softplus <- nn_softplus(beta = self$beta_sp, threshold = 20)

    flow_args <- c(d = self$d, device = self$device, flow_args)

    # Add flow parameters
    self$n_layers <- n_layers
    self$layers <- nn_module_list()

    for (i in 1:n_layers) {
      self$layers$append(do.call(flow_func, flow_args))
    }

    # Create forward method
    self$model <- nn_sequential(self$layers)

    # Add data to the model
    # Unsqueezing y to add a dimension - this enables broadcasting
    self$y <- y$to(device = self$device)$unsqueeze(2)
    self$x <- x$to(device = self$device)
    if (!self$mean_zero) {
      self$x_mean <- x_mean$to(device = self$device)
    }

    #create holders for prior a, c, lam and rate
    self$prior_a <- torch_tensor(a, device = self$device, requires_grad = FALSE)
    self$prior_c <- torch_tensor(c, device = self$device, requires_grad = FALSE)
    self$prior_a_mean <- torch_tensor(a_mean, device = self$device, requires_grad = FALSE)
    self$prior_c_mean <- torch_tensor(c_mean, device = self$device, requires_grad = FALSE)
    self$prior_rate <- torch_tensor(sigma2_rate, device = self$device, requires_grad = FALSE)
  },

  # Unnormalised log likelihood for Gaussian Process
  ldnorm = function(K, sigma2, beta) {
    n_latent <- sigma2$shape[1]

    single_eye <- torch_eye(self$N, device = self$device)$unsqueeze(1)
    batch_sigma2 <- single_eye$`repeat`(c(n_latent, 1, 1)) *
      sigma2$unsqueeze(2)$unsqueeze(2)

    L <- robust_chol(K + batch_sigma2, upper = FALSE)
    slogdet <- linalg_slogdet(K + batch_sigma2)

    if (self$mean_zero) {
      alpha <- torch_cholesky_solve(self$y$unsqueeze(1), L, upper = FALSE)
      log_lik <- -0.5 * slogdet[[2]] - 0.5 * torch_matmul(self$y$unsqueeze(1)$permute(c(1, 3, 2)), alpha)$squeeze()
    } else {
      y_demean <- (self$y - torch_matmul(self$x_mean, beta$t()))$t()$unsqueeze(3)
      alpha <- torch_cholesky_solve(y_demean, L, upper = FALSE)
      log_lik <- -0.5 * slogdet[[2]] - 0.5 * torch_matmul(y_demean$permute(c(1, 3, 2)), alpha)$squeeze()
    }

    return(log_lik)
  },

  # Unnormalised log density of triple gamma prior
  ltg = function(x, a, c, lam) {
    res <-  0.5 * torch_log(lam$unsqueeze(2)) -
      0.5 * torch_log(x) +
      log_hyperu(c + 0.5, 1.5 - a, a* x/(4.0 * c) * lam$unsqueeze(2))

    return(res)
  },

  # Unnormalised log density of normal-gamma-gamma prior
  ngg = function(x, a, c, lam) {
    res <- 0.5 * torch_log(lam$unsqueeze(2)) +
      log_hyperu(c + 0.5, 1.5 - a,  a * x^2/(4.0 * c) * lam$unsqueeze(2))

    return(res)
  },

  # Unnormalised log density of exponential distribution
  lexp = function(x, rate) {
    return(torch_log(rate) - rate * x)
  },

  # Unnormalised log density of F distribution
  ldf = function(x, d1, d2) {
    res <- (d1 * 0.5 - 1.0) * torch_log(x) - (d1 + d2) * 0.5 *
      torch_log1p(d1 / d2 * x)

    return(res)
  },

  # Forward method for GPR
  forward = function(zk) {
    log_det_J <- 0

    for (layer in 1:self$n_layers) {
      layer_out <- self$layers[[layer]]$forward(zk)
      log_det_J <- log_det_J + layer_out$log_diag_j
      zk <- layer_out$zk
    }

    # logdet jacobian of softplus transformation
    if (self$mean_zero) {
      log_det_J <- log_det_J + self$beta_sp * torch_sum(zk - self$softplus(zk), dim = 2)
      zk <- self$softplus(zk)
    } else {
      log_det_J <- log_det_J + self$beta_sp * torch_sum(zk[, 1:(self$x$shape[2] + 2)] - self$softplus(zk[, 1:(self$x$shape[2] + 2)]), dim = 2)
      l2_sigma_lam <- self$softplus(zk[, 1:(self$x$shape[2] + 2)])

      log_det_J <- log_det_J + self$beta_sp * (zk[, -1] - self$softplus(zk[, -1]))
      lam_mean <- self$softplus(zk[, -1])

      zk <- torch_cat(list(l2_sigma_lam, zk[, (self$x$shape[2] + 2):(self$x$shape[2] + 2 + self$x_mean$shape[2])], lam_mean$unsqueeze(2)), dim = 2)
    }

    return(list(zk = zk, log_det_J = log_det_J))
  },

  gen_batch = function(n_latent) {
    # Generate a batch of samples from the model
    z <- torch_randn(n_latent, self$d, device = self$device)
    return(z)
  },

  elbo = function(zk_pos, log_det_J) {
    # Extract the components of the variational distribution
    # Convention:
    # First x$shape[2] components are the theta parameters
    # Next component is the sigma parameter
    # Next component is the lambda parameter
    # Next x_mean$shape[2] components are the mean parameters
    # Last component is the lambda parameter for the mean
    l2_zk <- zk_pos[, 1:self$x$shape[2]]
    sigma_zk <- zk_pos[, (self$x$shape[2] + 1)]
    lam_zk <- zk_pos[, (self$x$shape[2] + 2)]

    if (!self$mean_zero) {
      beta <- zk_pos[, (self$x$shape[2] + 3):(self$x$shape[2] + 2 + self$x_mean$shape[2])]
      lam_mean <- zk_pos[, -1]
    } else {
      beta <- NULL
    }

    # Calculate covariance matrix
    K <- self$kernel_func(l2_zk, lam_zk, self$x)

    # Calculate the components of the ELBO
    likelihood <- self$ldnorm(K, sigma_zk, beta)$mean()

    prior <- self$ltg(l2_zk, self$prior_a, self$prior_c, lam_zk)$sum(dim = 2)$mean() +
      self$ldf(lam_zk/2, 2*self$prior_a, 2*self$prior_c)$mean() +
      self$lexp(sigma_zk, self$prior_rate)$mean()

    if (!self$mean_zero) {
      prior <- prior + self$ngg(beta, self$prior_a_mean, self$prior_c_mean, lam_zk)$sum(dim = 2)$mean() +
        self$ldf(lam_mean/2, 2*self$prior_a_mean, 2*self$prior_c_mean)$mean()
    }

    var_dens <- log_det_J$mean()

    # Compute ELBO
    elbo <- likelihood + prior + var_dens

    if (torch_isnan(elbo)$item()) {
      stop("ELBO is NaN")
    }

    return(elbo)
  }
)
