# Create nn_module subclass that implements forward methods for GPR
MVGPR_class <- nn_module(
  classname = "MVGPR",
  initialize = function(y,
                        x,
                        a = 0.5,
                        c = 0.5,
                        eta = 4,
                        a_Om = 0.5,
                        c_Om = 0.5,
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
    # Dimension of D + dimension of S-1 + dimension of theta + tau + tau_Om + sigma2
    self$dim <- self$M * (self$M - 1) / 2 + self$M - 1 + self$d + 1 + 1 + 1

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
    self$y <- nn_buffer(y$to(device = self$device))
    self$x <- nn_buffer(x$to(device = self$device))

    # #create holders for prior a, c, lam and rate
    self$prior_a <- nn_buffer(torch_tensor(a, device = self$device, requires_grad = FALSE))
    self$prior_c <- nn_buffer(torch_tensor(c, device = self$device, requires_grad = FALSE))
    self$prior_eta <- nn_buffer(torch_tensor(eta, device = self$device, requires_grad = FALSE))
    self$prior_a_Om <- nn_buffer(torch_tensor(a_Om, device = self$device, requires_grad = FALSE))
    self$prior_c_Om <- nn_buffer(torch_tensor(c_Om, device = self$device, requires_grad = FALSE))
    self$prior_rate <- nn_buffer(torch_tensor(sigma2_rate, device = self$device, requires_grad = FALSE))
  },

  # Unnormalised log likelihood for MV Gaussian Process
  ldnorm = function(K, L_Om, sigma2) {
    n_latent <- K$size(1)
    I <- torch_eye(self$N, device=self$device)$unsqueeze(1)$expand(c(n_latent, self$N, self$N))
    K_eps <- K + I * sigma2$view(c(n_latent, 1, 1))
    L_K <- robust_chol(K_eps, upper = FALSE)
    .shrinkGPR_internal$jit_funcs$ldnorm_multi(L_K, L_Om, self$y, as.integer(self$M), as.integer(self$N))
  }
  ,

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
      log_hyperu(c + 0.5, 1.5 - a,  a * x$pow(2)/(4.0 * c) * lam$unsqueeze(2))

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

  # Stan-style inverse transform: y (unconstrained) -> L (Cholesky of corr) and log|J|
  make_corr_chol = function(Omega_uncons) {

    # Omega_uncons: (n_latent, M*(M-1)/2)
    n_latent <- Omega_uncons$size(1)

    # z = tanh(Omega_uncons) in (-1, 1)
    pOmega <- self$M * (self$M - 1) / 2
    temp <- 2 + 4 * sqrt(log(pOmega))
    u <- Omega_uncons / temp
    z_vec <- torch_tanh(u)

    # Pack z_vec into strictly-lower-triangular matrix z_mat, filled by row
    z_mat <-  .shrinkGPR_internal$jit_funcs$make_tril(z_vec, self$M)

    L <- torch_zeros(c(n_latent, self$M, self$M), device = Omega_uncons$device)

    L_row <- torch_zeros(c(n_latent, self$M), device = Omega_uncons$device)
    L_row[, 1] <- 1
    L[, 1, ] <- L_row

    # Jacobian pieces:
    # 1) tanh part: sum log(1/cosh(Omega_uncons)^2) = -2 * sum log cosh(Omega_uncons)
    # Stable logcosh(Omega_uncons) = log( exp(Omega_uncons)+exp(-Omega_uncons) ) - log(2)
    log2 <- torch_log(torch_tensor(2.0, device = Omega_uncons$device))
    logcosh <- torch_logaddexp(u, -u) - log2
    logJ_tanh <- -2.0 * torch_sum(logcosh, dim = 2) - pOmega * torch_log(torch_tensor(temp, device = Omega_uncons$device))

    # 2) stick-breaking part: 0.5 * sum_{i>j} log(rem_{i,j})
    logJ_sb <- torch_zeros(c(n_latent), device = Omega_uncons$device)

    if (self$M >= 2) {
      for (i in 2:self$M) {
        rem <- torch_ones(c(n_latent), device = Omega_uncons$device)
        L_row <- torch_zeros(c(n_latent, self$M), device = Omega_uncons$device)

        for (j in 1:(i - 1)) {
          rem_before <- torch_clamp(rem, min = 1e-3)
          val <- z_mat[, i, j] * torch_sqrt(rem_before)
          L_row[, j] <- val
          rem <- rem_before - val$pow(2)

          logJ_sb <- logJ_sb + 0.5 * torch_log(rem_before)
        }

        L_row[, i] <- torch_sqrt(torch_clamp(rem, min = 1e-3))
        L[, i, ] <- L_row
      }
    }

    logJ <- logJ_tanh + logJ_sb
    return(list(L = L, logJ = logJ))
  },


  # Forward method for MVGPR
  forward = function(zk) {
    log_det_J <- 0

    for (layer in 1:self$n_layers) {
      layer_out <- self$layers[[layer]]$forward(zk)
      log_det_J <- log_det_J + layer_out$log_diag_j
      zk <- layer_out$zk
    }

    # Unconstrained elements of Omega are not restrained to be positive
    omega_comp <- self$M * (self$M - 1) / 2 + self$M - 1

    # All others are positive
    log_det_J <- log_det_J + self$beta_sp * torch_sum(zk[, (omega_comp + 1):self$dim] - self$softplus(zk[, (omega_comp + 1):self$dim]), dim = 2)
    non_omega <- self$softplus(zk[, (omega_comp + 1):self$dim])

    zk <- torch_cat(list(zk[, 1:omega_comp],
                         non_omega),
                    dim = 2)

    return(list(zk = zk, log_det_J = log_det_J))
  },

  gen_batch = function(n_latent) {
    # Generate a batch of samples from the model
    z <- torch_randn(n_latent, self$dim, device = self$device)

    # Specifically scale down the Omega components to push closer to identity
    # This stabilizes training, particularly in higher dimensions and early on
    omega_comp <- self$M * (self$M - 1) / 2 + self$M - 1
    scale_omega <- 0.2 / sqrt(omega_comp)
    z[, 1:omega_comp] <- scale_omega * z[, 1:omega_comp]

    theta_start <- omega_comp + 1
    theta_end <- omega_comp + self$d
    # push theta block negative so softplus(theta) starts near small values
    z[, theta_start:theta_end] <- z[, theta_start:theta_end] - 8

    return(z)
  },

  elbo = function(zk_pos, log_det_J) {
    # Extract the components of the variational distribution
    # Convention:
    # First (self$M * (self$M - 1) / 2) - self$M components are the unconstrained parameters of
    # the correlation matrix D
    # Next M-1 are the parameters for the scale vector of Omega (matrix S)
    # Next self$d components are the theta parameters for kernel that generates K
    # Next component is the tau parameter (glob shrinkage for theta)
    # Next component is tau_Om parameter (glob shrinkage for Omega)
    # Next component is sigma2 parameter
    omega_comp <- self$M * (self$M - 1) / 2 + self$M - 1

    D_uncons <-  zk_pos[, 1:(self$M * (self$M - 1) / 2)]
    S_uncons <- zk_pos[, (self$M * (self$M - 1) / 2 + 1):omega_comp]
    theta_zk <- zk_pos[, (omega_comp + 1):(omega_comp + self$d)]
    tau_zk <- zk_pos[, (omega_comp + self$d + 1)]
    tau_Om_zk <- zk_pos[, (omega_comp + self$d + 2)]
    sigma_zk <- zk_pos[, (omega_comp + self$d + 3)]

    # Res protector to avoid issues with 0 values
    theta_zk <- res_protector_autograd(theta_zk)
    tau_zk <- res_protector_autograd(tau_zk, tol = 1e-4)
    sigma_zk <- res_protector_autograd(sigma_zk, tol = 1e-4)

    # Calculate covariance matrix Sigma
    K <- self$kernel_func(theta_zk, tau_zk, self$x)
    if (torch_isnan(K$mean())$item()) {
      cat(sprintf("K is NaN. Check components:\ntheta_zk: %g\ntau_zk: %g\n",
                  theta_zk$mean()$item(), tau_zk$mean()$item()))
      stop("K is NaN")
    }

    # Calculate cholesky of correlation matrix D
    D_chol_zk <- self$make_corr_chol(D_uncons)

    # Smooth bound of S_uncons to improve stability of likelihood calculation, particularly early on and in higher dimensions
    b <- 4
    S_uncons_c <- b * torch_tanh(S_uncons / b)

    S_M <- -torch_sum(S_uncons_c, dim=2, keepdim=TRUE)
    S_logdiag <- torch_cat(list(S_uncons_c, S_M), dim=2)

    S_diag <- torch_exp(S_logdiag)
    S <- torch_diag_embed(S_diag)

    # Calculate log determinant of smooth bound, which has two components: the exp map and the squashing from unconstrained to constrained space
    log_det_exp <- torch_sum(S_uncons_c, dim=2)
    log_det_squash <- torch_sum(torch_log(torch_clamp(1 / torch_cosh(S_uncons / b)$pow(2), min=1e-12)), dim=2)
    log_det_S <- log_det_exp + log_det_squash

    # Calculate cholesky of Omega
    L_Om <- torch_bmm(S, D_chol_zk$L)

    # Slightly bias diagonal of L_Om away from zero to improve stability of likelihood calculation
    eps_diag <- 0.03
    beta <- 10
    diag <- torch_diagonal(L_Om, dim1=2, dim2=3)
    diag2 <- eps_diag + nnf_softplus(diag - eps_diag, beta = beta)

    L_Om2 <- L_Om$clone()
    L_Om2$diagonal(dim1=2, dim2=3)$copy_(diag2)



    likelihood <- self$ldnorm(K, L_Om2, sigma_zk)$mean()
    if (torch_isnan(likelihood)$item()) {
      stop("Likelihood is NaN")
    }

    diag_LD  <- torch_diagonal(D_chol_zk$L, dim1=2, dim2=3)

    logdet_D <- 2 * torch_sum(torch_log(diag_LD), dim=2)
    lkj_term <- (self$prior_eta - 1) * logdet_D

    # Prior on theta
    prior <- self$ltg(theta_zk, self$prior_a, self$prior_c, tau_zk)$sum(dim = 2)$mean() +
      self$ldf(tau_zk/2, 2*self$prior_c, 2*self$prior_a)$mean() +
      # Prior on D (LKJ)
      lkj_term$mean() +
      # Prior on S
      self$ltg(S_diag, self$prior_a_Om, self$prior_c_Om, tau_Om_zk)$sum(dim = 2)$mean() +
      # Prior on tau_Om
      self$ldf(tau_Om_zk/2, 2*self$prior_c_Om, 2*self$prior_a_Om)$mean() +
      # Prior on sigma^2
      self$lexp(sigma_zk, self$prior_rate)$mean()

    var_dens <- log_det_J$mean() + D_chol_zk$logJ$mean() + log_det_S$mean()

    # Compute ELBO
    elbo <- likelihood + prior + var_dens

    if (torch_isnan(elbo)$item()) {
      stop("ELBO is NaN")
    }

    return(elbo)
  },

  # Method to calculate moments of predictive distribution
  calc_pred_moments = function(x_new, nsamp) {

    with_no_grad({
      N_new = x_new$size(1)

      # First, generate posterior draws by drawing random samples from the variational distribution.
      z <- self$gen_batch(nsamp)
      zk_pos <- self$forward(z)$zk

      # Extract the components of the variational distribution
      # Convention:
      # First (self$M * (self$M - 1) / 2) - self$M components are the unconstrained parameters of
      # the correlation matrix D
      # Next M-1 are the parameters for the scale vector of Omega (matrix S)
      # Next self$d components are the theta parameters for kernel that generates K
      # Next component is the tau parameter (glob shrinkage for theta)
      # Next component is tau_Om parameter (glob shrinkage for Omega)
      # Next component is sigma2 parameter

      omega_comp <- self$M * (self$M - 1) / 2 + self$M - 1

      D_uncons <-  zk_pos[, 1:(self$M * (self$M - 1) / 2)]
      S_uncons <- zk_pos[, (self$M * (self$M - 1) / 2 + 1):omega_comp]
      theta_zk <- zk_pos[, (omega_comp + 1):(omega_comp + self$d)]
      tau_zk <- zk_pos[, (omega_comp + self$d + 1)]
      tau_Om_zk <- zk_pos[, (omega_comp + self$d + 2)]
      sigma_zk <- zk_pos[, (omega_comp + self$d + 3)]

      # Res protector to avoid issues with 0 values
      theta_zk <- res_protector_autograd(theta_zk)
      tau_zk <- res_protector_autograd(tau_zk, tol = 1e-4)
      sigma_zk <- res_protector_autograd(sigma_zk)

      # Calculate covariance matrix Sigma
      K <- self$kernel_func(theta_zk, tau_zk, self$x)

      # Calculate cholesky of correlation matrix D
      D_chol_zk <- self$make_corr_chol(D_uncons)

      # Smooth bound of S_uncons to improve stability of likelihood calculation, particularly early on and in higher dimensions
      b <- 4
      S_uncons_c <- b * torch_tanh(S_uncons / b)

      S_M <- -torch_sum(S_uncons_c, dim=2, keepdim=TRUE)
      S_logdiag <- torch_cat(list(S_uncons_c, S_M), dim=2)

      S_diag <- torch_exp(S_logdiag)
      S <- torch_diag_embed(S_diag)

      # Calculate log determinant of smooth bound, which has two components: the exp map and the squashing from unconstrained to constrained space
      log_det_exp <- torch_sum(S_uncons_c, dim=2)
      log_det_squash <- torch_sum(torch_log(torch_clamp(1 / torch_cosh(S_uncons / b)$pow(2), min=1e-12)), dim=2)
      log_det_S <- log_det_exp + log_det_squash

      # Calculate cholesky of Omega
      L_Om <- torch_bmm(S, D_chol_zk$L)

      # Slightly bias diagonal of L_Om away from zero to improve stability of likelihood calculation
      eps_diag <- 0.03
      beta <- 10
      diag <- torch_diagonal(L_Om, dim1=2, dim2=3)
      diag2 <- eps_diag + nnf_softplus(diag - eps_diag, beta = beta)

      L_Om2 <- L_Om$clone()
      L_Om2$diagonal(dim1=2, dim2=3)$copy_(diag2)
      Omega <- torch_bmm(L_Om2, L_Om2$permute(c(1, 3, 2)))

      # Transform covariance matrix K and transform into L and alpha
      # L is the cholseky decomposition of K + sigma^2I, i.e. the covariance matrix of the GP
      # alpha is the solution to L L^T alpha = y, i.e. (K + sigma^2I)^{-1}y
      single_eye <- torch_eye(self$N, device = self$device)
      batch_sigma2 <- single_eye$`repeat`(c(nsamp, 1, 1)) *
        sigma_zk$unsqueeze(2)$unsqueeze(2)
      L <- robust_chol(K + batch_sigma2, upper = FALSE)

      alpha <- torch_cholesky_solve(self$y, L, upper = FALSE)


      # Calculate K_star_star, the covariance between the test data
      K_star_star <- self$kernel_func(theta_zk, tau_zk, x_new)

      # Calculate K_star, the covariance between the training and test data
      K_star_t <- self$kernel_func(theta_zk, tau_zk, self$x, x_new)

      # Calculate the predictive mean and variance
      pred_mean <- torch_bmm(K_star_t, alpha)


      single_eye_new <- torch_eye(N_new, device = self$device)
      batch_sigma2_new <- single_eye_new$`repeat`(c(nsamp, 1, 1)) *
        sigma_zk$unsqueeze(2)$unsqueeze(2)
      v <- linalg_solve_triangular(L, K_star_t$permute(c(1, 3, 2)), upper = FALSE)
      K_post <- K_star_star - torch_matmul(v$permute(c(1, 3, 2)), v) + batch_sigma2_new

      return(list(pred_mean = pred_mean, K = K_post, Omega = Omega))
    })
  },

  predict = function(x_new, nsamp) {

    with_no_grad({
      N_new <- x_new$size(1)

      # Calculate the moments of the predictive distribution
      pred_moments <- self$calc_pred_moments(x_new, nsamp)
      pred_mean <- pred_moments$pred_mean
      pred_K <- pred_moments$K
      pred_Omega <- pred_moments$Omega

      L_S <- robust_chol(pred_K)
      L_Om <- robust_chol(pred_Omega)

      # Slightly bias diagonal of L_Om away from zero to improve stability of likelihood calculation
      eps_diag <- 0.03
      beta <- 10
      diag <- torch_diagonal(L_Om, dim1=2, dim2=3)
      diag2 <- eps_diag + nnf_softplus(diag - eps_diag, beta = beta)

      L_Om2 <- L_Om$clone()
      L_Om2$diagonal(dim1=2, dim2=3)$copy_(diag2)

      Z <- torch_randn(c(nsamp, N_new, self$M), device=self$device)
      pred_samples <- pred_mean + torch_bmm(torch_bmm(L_S, Z), L_Om2$permute(c(1, 3, 2)))

      return(pred_samples)
    })

  },

  # Method to evaluate predictive density
  eval_pred_dens = function(y_new, x_new, nsamp, log = FALSE) {

    with_no_grad({
      n_eval <- y_new$size(1)
      M <- y_new$size(2)

      pred_moments <- self$calc_pred_moments(x_new, nsamp)
      pred_mean <- pred_moments$pred_mean
      pred_K <- pred_moments$K
      pred_Omega <- pred_moments$Omega



      # Build diff for all y's and all draws:
      diff <- y_new - pred_mean

      L_Om <- robust_chol(pred_Omega, upper = FALSE)

      X <- linalg_solve_triangular(L_Om, diff$permute(c(1,3,2)), upper = FALSE)

      quad_omega <- torch_sum(X$pow(2), dim = 2)
      quad <- quad_omega / pred_K$squeeze(3)

      L_K <- robust_chol(pred_K, upper = FALSE)

      # Compute log determinant
      diag_L_Om <- torch_diagonal(L_Om, dim1 = 2, dim2 = 3)
      logdet_Om <- 2 * torch_log(diag_L_Om)$sum(dim = 2)

      logdet_K <- M * torch_log(pred_K)


      log_comp <- -0.5 * (quad + M * log(2 * pi) + logdet_K$squeeze(3) + logdet_Om$unsqueeze(2))

      m <- torch_max(log_comp, dim = 1)[[1]]
      res <- m + torch_log(torch_mean(torch_exp(log_comp - m$unsqueeze(1)), dim = 1))

      if (!log) {
        res <- torch_exp(res)
      }

      return(res)
    })

  }
)
