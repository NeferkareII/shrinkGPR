MVTPR_class <- nn_module(
  classname = "MVTPR",
  inherit = MVGPR_class,
  initialize = function(y,
                        x,
                        a = 0.5,
                        c = 0.5,
                        eta = 4,
                        a_Om = 0.5,
                        c_Om = 0.5,
                        sigma2_rate = 10,
                        nu_alpha = 0.5,
                        nu_beta = 2,
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
    # Dimension of D + dimension of S-1 + dimension of theta + tau + tau_Om + sigma2 + nu
    self$dim <- self$M * (self$M - 1) / 2 + self$M - 1 + self$d + 1 + 1 + 1 + 1

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

    # self$parameters <- 1/sqrt(n_layers) * self$parameters

    # Create forward method
    self$model <- nn_sequential(self$layers)

    # Add data to the model
    # Unsqueezing y to add a dimension - this enables broadcasting
    self$y <- y$to(device = self$device)
    self$x <- x$to(device = self$device)

    #create holders for prior a, c, lam and rate
    self$prior_a <- torch_tensor(a, device = self$device, requires_grad = FALSE)
    self$prior_c <- torch_tensor(c, device = self$device, requires_grad = FALSE)
    self$prior_eta <- torch_tensor(eta, device = self$device, requires_grad = FALSE)
    self$prior_a_Om <- torch_tensor(a_Om, device = self$device, requires_grad = FALSE)
    self$prior_c_Om <- torch_tensor(c_Om, device = self$device, requires_grad = FALSE)
    self$prior_rate <- torch_tensor(sigma2_rate, device = self$device, requires_grad = FALSE)

    # For prior on nu
    self$nu_alpha <- torch_tensor(nu_alpha, device = self$device, requires_grad = FALSE)
    self$nu_beta <- torch_tensor(nu_beta, device = self$device, requires_grad = FALSE)
  },

  ldg = function(x, alpha, beta) {
    res <- (alpha - 1.0) * torch_log(x) - beta * x
    return(res)
  },

  ldt = function(K, L_Om, sigma2, nu) {
    n_latent <- K$size(1)

      I_N  <- torch_eye(self$N, device=self$device)$unsqueeze(1)$expand(c(n_latent, self$N, self$N))
    K_eps <- K + I_N * sigma2$view(c(n_latent, 1, 1))
    L_K <- robust_chol(K_eps, upper = FALSE)

    # Expand Y for batching: (n_latent, N, M)
    Y <- self$y$unsqueeze(1)$expand(c(n_latent, self$N, self$M))
    Yt <- Y$transpose(-2, -1)

    # B = Y^T K^{-1} Y via cholesky_solve
    alpha <- torch_cholesky_solve(Y, L_K, upper = FALSE)
    B <- torch_bmm(Yt, alpha)

    # X = L_Om^{-1} B
    X <- linalg_solve_triangular(L_Om, B, upper = FALSE, left = TRUE)

    # C = X L_Om^{-T}  <=>  C (L_Om^T) = X  (right-side triangular solve)
    C <- linalg_solve_triangular(L_Om$transpose(-2, -1), X, upper = TRUE, left = FALSE)

    # logdet(I + C)
    I_M <- torch_eye(self$M, device=self$device)$unsqueeze(1)$expand(c(n_latent, self$M, self$M))
    L_Iplus <- robust_chol(I_M + C, upper = FALSE)
    diag_LI <- torch_diagonal(L_Iplus, dim1 = -2, dim2 = -1)
    ld_Iplus <- 2 * torch_sum(torch_log(diag_LI), dim = 2)

    # logdet(K) and logdet(Omega) from Cholesky factors
    diag_K  <- torch_diagonal(L_K,  dim1 = -2, dim2 = -1)
    diag_Om <- torch_diagonal(L_Om, dim1 = -2, dim2 = -1)
    ld_K  <- 2 * torch_sum(torch_log(diag_K),  dim = 2)
    ld_Om <- 2 * torch_sum(torch_log(diag_Om), dim = 2)

    # NLL
    nll <- 0.5 * (nu + self$N + self$M - 1) * ld_Iplus +
      0.5 * self$M * ld_K +
      0.5 * self$N * ld_Om +
      ( torch_mvlgamma(0.5 * (nu + self$N - 1), self$N) -
          torch_mvlgamma(0.5 * (nu + self$N + self$M - 1), self$N) )

    # Return log-likelihood
    -nll

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
    # Next component is nu parameter
    omega_comp <- self$M * (self$M - 1) / 2 + self$M - 1

    D_uncons <-  zk_pos[, 1:(self$M * (self$M - 1) / 2)]
    S_uncons <- zk_pos[, (self$M * (self$M - 1) / 2 + 1):omega_comp]
    theta_zk <- zk_pos[, (omega_comp + 1):(omega_comp + self$d)]
    tau_zk <- zk_pos[, (omega_comp + self$d + 1)]
    tau_Om_zk <- zk_pos[, (omega_comp + self$d + 2)]
    sigma_zk <- zk_pos[, (omega_comp + self$d + 3)]
    nu_zk <- zk_pos[, (omega_comp + self$d + 4)]

    # Res protector to avoid issues with 0 values
    theta_zk <- res_protector_autograd(theta_zk)
    tau_zk <- res_protector_autograd(tau_zk, tol = 1e-4)
    sigma_zk <- res_protector_autograd(sigma_zk, tol = 1e-4)

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

    likelihood <- self$ldt(K, L_Om2, sigma_zk, nu_zk)$mean()

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
      self$lexp(sigma_zk, self$prior_rate)$mean() +
      # Prior on nu
      self$ldg(nu_zk, self$nu_alpha, self$nu_beta)$mean()

    var_dens <- log_det_J$mean() + D_chol_zk$logJ$mean() + log_det_S$mean()

    # Compute ELBO
    elbo <- likelihood + prior + var_dens

    if (torch_isnan(elbo)$item()) {
      stop("ELBO is NaN")
    }

    return(elbo)
  },

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
      # Next component is nu parameter

      omega_comp <- self$M * (self$M - 1) / 2 + self$M - 1

      D_uncons <-  zk_pos[, 1:(self$M * (self$M - 1) / 2)]
      S_uncons <- zk_pos[, (self$M * (self$M - 1) / 2 + 1):omega_comp]
      theta_zk <- zk_pos[, (omega_comp + 1):(omega_comp + self$d)]
      tau_zk <- zk_pos[, (omega_comp + self$d + 1)]
      tau_Om_zk <- zk_pos[, (omega_comp + self$d + 2)]
      sigma_zk <- zk_pos[, (omega_comp + self$d + 3)]
      nu_zk <- zk_pos[, (omega_comp + self$d + 4)]

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

      # Transform covariance matrix K into L and alpha
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

      Omega_hat <- Omega + torch_bmm(self$y$t()$unsqueeze(1)$expand(c(nsamp, self$M, self$N)), alpha)
      nu_hat <- nu_zk + self$N

      return(list(pred_mean = pred_mean, K = K_post, Omega = Omega_hat, nu = nu_hat))
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
      pred_nu <- pred_moments$nu

      L_S <- robust_chol(pred_K)

      Z <- torch_randn(c(nsamp, N_new, self$M), device=self$device)

      # Posterior degrees of freedom: nu_hat = nu + N
      nu_hat <- pred_nu

      # Sample g ~ Gamma(nu_hat/2, nu_hat/2) per draw, so E[g] = 1
      # Then w = 1/g gives the inverse-chi-squared scaling
      # sqrt(w) applied to the Gaussian draws produces matrix-t samples
      g <- distr_gamma(
        concentration = (nu_hat / 2)$view(c(-1)),
        rate = torch_tensor(0.5, device = self$device)
      )$sample()

      w <- (1 / g)$view(c(-1, 1, 1))

      L_Om <- robust_chol(pred_Omega)
      pred_samples <- pred_mean +
        torch_sqrt(w) * torch_bmm(torch_bmm(L_S, Z), L_Om$permute(c(1, 3, 2)))

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
      pred_nu <- pred_moments$nu

      # Build diff for all y's and all draws:
      diff <- y_new - pred_mean

      L_K <- robust_chol(pred_K, upper = FALSE)
      L_Om <- robust_chol(pred_Omega, upper = FALSE)

      # Expand diff for batching: (n_latent, n_eval, M)
      Y <- diff$expand(c(nsamp, n_eval, M))
      Yt <- Y$transpose(-2, -1)

      # B = Y^T K^{-1} Y via cholesky_solve
      alpha <- torch_cholesky_solve(Y, L_K, upper = FALSE)
      B <- torch_bmm(Yt, alpha)

      # C = L_Om^{-1} B L_Om^{-T}
      X <- linalg_solve_triangular(L_Om, B, upper = FALSE, left = TRUE)
      C <- linalg_solve_triangular(L_Om$transpose(-2, -1), X, upper = TRUE, left = FALSE)

      # logdet(I + C)
      I_M <- torch_eye(M, device = self$device)$unsqueeze(1)$expand(c(nsamp, M, M))
      L_Iplus <- robust_chol(I_M + C, upper = FALSE)
      diag_LI <- torch_diagonal(L_Iplus, dim1 = -2, dim2 = -1)
      ld_Iplus <- 2 * torch_sum(torch_log(diag_LI), dim = 2)

      # logdet(K) and logdet(Omega) from Cholesky factors
      diag_K  <- torch_diagonal(L_K,  dim1 = -2, dim2 = -1)
      diag_Om <- torch_diagonal(L_Om, dim1 = -2, dim2 = -1)
      ld_K  <- 2 * torch_sum(torch_log(diag_K),  dim = 2)
      ld_Om <- 2 * torch_sum(torch_log(diag_Om), dim = 2)

      # NLL
      nll <- 0.5 * (pred_nu + n_eval + M - 1) * ld_Iplus +
        0.5 * M * ld_K +
        0.5 * n_eval * ld_Om +
        0.5 * n_eval * M * log(pi) +
        ( torch_mvlgamma(0.5 * (pred_nu + n_eval - 1), n_eval) -
            torch_mvlgamma(0.5 * (pred_nu + n_eval + M - 1), n_eval) )

      log_comp <- -nll

      m <- torch_max(log_comp, dim = 1)[[1]]
      res <- m + torch_log(torch_mean(torch_exp(log_comp - m$unsqueeze(1)), dim = 1))

      if (!log) {
        res <- torch_exp(res)
      }

      return(res)
    })
  }
)

