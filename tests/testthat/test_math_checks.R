fit_univariate <- function(fit_func) {
  set.seed(123)
  torch_manual_seed(123)

  dat <- data.frame(
    y = sin(2 * pi * runif(20)) + rnorm(20, sd = 0.1),
    x1 = runif(20),
    x2 = rnorm(20)
  )

  res <- fit_func(y ~ x1 + x2, data = dat[1:15, ], n_epochs = 10, display_progress = FALSE)
  list(res = res, test = dat[16:20, ])
}

test_that("predictive samples match predictive moments at a single point", {
  if (!torch::torch_is_installed()) {
    skip("Torch is not installed. Skipping test.")
  }

  for (fit_func in list(shrinkGPR, shrinkTPR)) {
    fit <- fit_univariate(fit_func)
    newdata <- fit$test[1, ]
    nsamp <- 20000

    preds <- predict(fit$res, newdata = newdata, nsamp = nsamp)
    expect_equal(dim(preds), c(nsamp, 1))

    # Variance of the predictive mixture: E[Var] + Var[E]
    # (for shrinkTPR the predictive scale is parameterized as a covariance)
    moments <- calc_pred_moments(fit$res, newdata = newdata, nsamp = nsamp)
    mix_mean <- mean(moments$means)
    mix_var <- mean(moments$vars) + var(as.vector(moments$means))

    expect_lt(abs(mean(preds) - mix_mean), 0.05 * sqrt(mix_var))
    expect_equal(var(as.vector(preds)), mix_var, tolerance = 0.1)
  }
})

test_that("simMVGPR returns Omega with the shrinkMVGPR normalization", {
  if (!torch::torch_is_installed()) {
    skip("Torch is not installed. Skipping test.")
  }

  set.seed(123)
  torch_manual_seed(123)

  # Default Omega
  sim <- simMVGPR(N = 20, M = 3)
  expect_equal(prod(sqrt(diag(sim$true_vals$Omega))), 1)

  # User supplied Omega is rescaled, but keeps its correlation structure
  Omega_in <- matrix(c(4, 1, 1, 9), 2, 2)
  sim <- simMVGPR(N = 20, M = 2, Omega = Omega_in)
  expect_equal(prod(sqrt(diag(sim$true_vals$Omega))), 1)
  expect_equal(cov2cor(sim$true_vals$Omega), cov2cor(Omega_in))
})
