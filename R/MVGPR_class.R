# Create nn_module subclass that implements forward methods for GPR
MVGPR_class <- nn_module(
  classname = "MVGPR",
  initialize = function(y,
                        x,
                        a = 0.5,
                        c = 0.5,
                        a_cov = 0.5,
                        c_cov = 0.5,
                        sigma2_rate = 10,
                        n_layers,
                        flow_func,
                        flow_args,
                        kernel_func = shrinkGPR::kernel_se,
                        device) {

    # Add dimension attributes
    self$d <- ncol(x)
    self$M <- ncol(y)
    self$N <- nrow(y)

    # Add atttribute for latent dimension
    # Dimension of Omega + kappa + dimension of theta + tau + sigma2
    self$dim <- self$M * (self$M + 1) / 2 + 1 + self$d + 1 + 1

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

    flow_args <- c(d = self$dim, flow_args)

    # Add flow parameters
    self$n_layers <- n_layers
    self$layers <- nn_module_list()

    for (i in 1:n_layers) {
      self$layers$append(do.call(flow_func, flow_args))
    }

    self$layers$to(device = self$device)

    # Create forward method
    self$model <- nn_sequential(self$layers)

    # Add data to the model
    # Unsqueezing y to add a dimension - this enables broadcasting
    self$y <- y$to(device = self$device)
    self$x <- x$to(device = self$device)

    #create holders for prior a, c, lam and rate
    self$prior_a <- torch_tensor(a, device = self$device, requires_grad = FALSE)
    self$prior_c <- torch_tensor(c, device = self$device, requires_grad = FALSE)
    self$prior_a_cov <- torch_tensor(a_cov, device = self$device, requires_grad = FALSE)
    self$prior_c_cov <- torch_tensor(c_cov, device = self$device, requires_grad = FALSE)
    self$prior_rate <- torch_tensor(sigma2_rate, device = self$device, requires_grad = FALSE)
  },

  # Unnormalised log likelihood for MV Gaussian Process
  ldnorm = function(K, Omega, sigma2) {
    n_latent <- K$shape[1]

    I <- torch_eye(self$N, device = self$device)$unsqueeze(1)$expand(c(n_latent, self$N, self$N))
    sigma_term = I * sigma2$view(c(n_latent, 1, 1))
    K_eps = K + sigma_term

    L_K <- torch_cholesky(K_eps, upper = FALSE)

    slogdet_L <- torch_slogdet(K_eps)[[2]]
    slogdet_Om <- torch_slogdet(Omega)[[2]]

    alpha_L <- torch_cholesky_solve(self$y, L_K, upper = FALSE)
    tr <- torch_sum(torch_diagonal(-0.5 * torch_bmm(
      torch_bmm(Omega, self$y$t()$expand(c(n_latent, self$M, self$N))),
      alpha_L),
      dim1=-2, dim2=-1),
      dim = 2)

    log_lik <- -0.5*self$M*slogdet_L + 0.5*self$N*slogdet_Om + tr

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

  # Forward method for MVGPR
  forward = function(zk) {
    log_det_J <- 0

    for (layer in 1:self$n_layers) {
      layer_out <- self$layers[[layer]]$forward(zk)
      log_det_J <- log_det_J + layer_out$log_diag_j
      zk <- layer_out$zk
    }

    # Diagonal components of cholesky of Omega are restrained to be positive
    log_det_J <- log_det_J + self$beta_sp * torch_sum(zk[, 1:self$M)] - self$softplus(zk[, 1:self$M]), dim = 2)
    omega_diag <- self$softplus(zk[, 1:self$M])

    # Off diagonal components of cholesky of Omega are not restrained to be positive
    omega_comp <- self$M * (self$M + 1) / 2

    # All others are positive
    log_det_J <- log_det_J + self$beta_sp * torch_sum(zk[, (omega_comp + 1):self$dim] - self$softplus(zk[, (omega_comp + 1):self$dim]), dim = 2)
    non_omega <- self$softplus(zk[, (omega_comp + 1):self$dim])

    zk <- torch_cat(list(omega_diag,
                         zk[, (self$M + 1):omega_comp],
                         non_omega),
                    dim = 2)

    return(list(zk = zk, log_det_J = log_det_J))
  },

  gen_batch = function(n_latent) {
    # Generate a batch of samples from the model
    z <- torch_randn(n_latent, self$dim, device = self$device)
    return(z)
  },

  elbo = function(zk_pos, log_det_J) {
    # Extract the components of the variational distribution
    # Convention:
    # First self$M components are the diagonal parameters of the cholesky factor of Omega
    # These have to be positive
    # Next (self$M * (self$M - 1) / 2) - self$M components are the off-diagonal parameters
    # of the cholesky factor of Omega
    # Next self$d components are the theta parameters for kernel that generates K
    # Next component is the kappa parameter (glob shrinkage for Omega)
    # Next component is the tau parameter (glob shrinkage for theta)
    # Next component is sigma2 parameter
    Omega_chol <-  zk_pos[, 1:(self$M * (self$M + 1) / 2)]
    theta_zk <- zk_pos[, (self$M * (self$M + 1) / 2 + 1):(self$M * (self$M + 1) / 2 + self$d)]
    kappa_zk <- zk_pos[, (self$M * (self$M + 1) / 2 + self$d + 1)]
    tau_zk <- zk_pos[, (self$M * (self$M + 1) / 2 + self$d + 2)]
    sigma_zk <- zk_pos[, (self$M * (self$M + 1) / 2 + self$d + 3)]

    # Calculate covariance matrix Sigma
    K <- self$kernel_func(theta_zk, tau_zk, self$x)

    # Calculate precision matrix by reconstructing from cholesky factor
    Omega_chol_zk <- .shrinkGPR_internal$jit_funcs$make_tril(Omega_chol, self$M)
    Omega <- torch_bmm(Omega_chol_zk, Omega_chol_zk$transpose(2, 3))

    # Hack for now
    # Omega[, 1, 1] <- 1


    # Calculate the components of the ELBO
    likelihood <- self$ldnorm(K, Omega, sigma_zk)$mean()

    # Get lower triangular part of Omega_chol_zk
    Omega_lower_diag <- Omega$tril(diagonal = -1)
    # Remove zeros
    Omega_lower_diag_vec <- Omega_lower_diag[Omega_lower_diag != 0]
    # Squeeze into correct shape
    Omega_lower_diag_zk <- Omega_lower_diag_vec$view(c(K$shape[1], -1))


    # Prior on theta
    prior <- self$ltg(theta_zk, self$prior_a, self$prior_c, tau_zk)$sum(dim = 2)$mean() +
      self$ldf(tau_zk/2, 2*self$prior_a, 2*self$prior_c)$mean() +
    # Prior on Omega (only off-diagonal elems)
      self$ngg(Omega_lower_diag_zk, self$prior_a_cov, self$prior_c_cov, kappa_zk)$sum(dim = 2)$mean() +
      self$ldf(kappa_zk/2, 2*self$prior_a_cov, 2*self$prior_c_cov)$mean() +
    # Trying exponential prior on diagonal elems of Omega
      #self$lexp(Omega$diagonal(dim1 = 2, dim2 = 3), torch_tensor(20, device = self$device))$sum(dim = 2)$mean() +
    # Prior on sigma^2
      self$lexp(sigma_zk, self$prior_rate)$mean()

    # Calculate log_det_J contribution of transformation to Omega
    diag_L <- Omega_chol[, 1:self$M]
    exponents <- torch_arange(self$M, 0, -1, device=self$device)
    log_jac_term <- self$M*torch_log(torch_tensor(2.0, device = self$device)) +
      (exponents$unsqueeze(1) * torch_log(diag_L))$sum(dim=2)


    var_dens <- log_det_J$mean() + log_jac_term$mean()

    # Compute ELBO
    elbo <- likelihood + prior + var_dens

    if (torch_isnan(elbo)$item()) {
      stop("ELBO is NaN")
    }

    return(elbo)
  },

  # Method to calculate moments of predictive distribution
  calc_pred_moments = function(x_new, nsamp, x_mean_new) {

    with_no_grad({
      N_new = x_new$shape[1]

      # First, generate posterior draws by drawing random samples from the variational distribution.
      z <- self$gen_batch(nsamp)
      zk_pos <- self$forward(z)$zk
      zk_pos <- res_protector_autograd(zk_pos)

      l2_zk <- zk_pos[, 1:self$x$shape[2]]
      sigma_zk <- zk_pos[, (self$x$shape[2] + 1)]
      lam_zk <- zk_pos[, (self$x$shape[2] + 2)]

      if (!self$mean_zero) {
        beta <- zk_pos[, (self$x$shape[2] + 3):(self$x$shape[2] + 2 + self$x_mean$shape[2])]
      } else {
        beta <- NULL
      }

      # Calculate covariance matrix K and transform into L and alpha
      # L is the cholseky decomposition of K + sigma^2I, i.e. the covariance matrix of the GP
      # alpha is the solution to L L^T alpha = y, i.e. (K + sigma^2I)^{-1}y

      K <- self$kernel_func(l2_zk, lam_zk, self$x)
      single_eye <- torch_eye(self$N, device = self$device)
      batch_sigma2 <- single_eye$`repeat`(c(nsamp, 1, 1)) *
        sigma_zk$unsqueeze(2)$unsqueeze(2)
      L <- robust_chol(K + batch_sigma2, upper = FALSE)

      if (self$mean_zero) {
        alpha <- torch_cholesky_solve(self$y, L, upper = FALSE)
      } else {
        y_demean <- (self$y - torch_matmul(self$x_mean, beta$t()))$t()$unsqueeze(3)
        alpha <- torch_cholesky_solve(y_demean, L, upper = FALSE)
      }

      # Calculate K_star_star, the covariance between the test data
      K_star_star <- self$kernel_func(l2_zk, lam_zk, x_new)

      # Calculate K_star, the covariance between the training and test data
      K_star_t <- self$kernel_func(l2_zk, lam_zk, self$x, x_new)

      # Calculate the predictive mean and variance
      if (self$mean_zero) {
        pred_mean <- torch_bmm(K_star_t, alpha)$squeeze()
      } else {
        pred_mean <- torch_bmm(K_star_t, alpha)$squeeze() +
          torch_matmul(x_mean_new, beta$t())$t()$squeeze()
      }

      single_eye_new <- torch_eye(N_new, device = self$device)
      batch_sigma2_new <- single_eye_new$`repeat`(c(nsamp, 1, 1)) *
        sigma_zk$unsqueeze(2)$unsqueeze(2)
      v <- linalg_solve_triangular(L, K_star_t$permute(c(1, 3, 2)), upper = FALSE)
      pred_var <- K_star_star - torch_matmul(v$permute(c(1, 3, 2)), v) + batch_sigma2_new

      return(list(pred_mean = pred_mean, pred_var = pred_var))
    })
  },

  predict = function(x_new, nsamp, x_mean_new) {

    with_no_grad({
      N_new <- x_new$shape[1]

      # Calculate the moments of the predictive distribution
      pred_moments <- self$calc_pred_moments(x_new, nsamp, x_mean_new)
      pred_mean <- pred_moments$pred_mean
      pred_var <- pred_moments$pred_var

      pred_var_chol <- robust_chol(pred_var, upper = FALSE)
      eps <- torch_randn(nsamp, N_new, 1, device = self$device)

      pred_samples <- pred_mean$unsqueeze(1) + torch_bmm(pred_var_chol, eps)$squeeze()

      return(pred_samples$squeeze())
    })

  },

  # Method to evaluate predictive density
  eval_pred_dens = function(y_new, x_new, nsamp, x_mean_new = NULL, log = FALSE) {

    with_no_grad({
      pred_moments <- self$calc_pred_moments(x_new, nsamp, x_mean_new)
      pred_mean <- pred_moments$pred_mean
      pred_var <- pred_moments$pred_var$squeeze()

      ldnorm <- distr_normal(pred_mean, torch_sqrt(pred_var))

      log_dens <- ldnorm$log_prob(y_new$unsqueeze(2))$t()
      max <- torch_max(log_dens, dim = 1)
      res <- -torch_log(torch_tensor(nsamp, device = self$device)) + max[[1]] + torch_log(torch_sum(torch_exp(log_dens - max[[1]]$unsqueeze(1)), dim = 1))

      if (!log) {
        res <- torch_exp(res)
      }

      return(res)
    })

  },

  # Method to calculate LPDS
  LPDS = function(x_new, y_new, nsamp, x_mean_new = NULL) {
    res <- self$eval_pred_dens(x_new, y_new, nsamp, x_mean_new, log = TRUE)
    return(res)
  }

)
