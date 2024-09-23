# Function that returns pvalue for simple linear regression
lightweight_ols <- function(y, x) {
  ym <- mean(y)
  xm <- mean(x)

  beta_hat <- sum((x - xm) * (y - ym)) / sum((x - xm)^2)
  alpha_hat <- ym - beta_hat * xm

  resids <- y - alpha_hat - beta_hat * x
  sigma2_hat <- sum(resids^2) / (length(y) - 2)

  beta_hat_var <- sigma2_hat / sum((x - xm)^2)
  alpha_hat_var <- sigma2_hat * (1 / length(y) + xm^2 / sum((x - xm)^2))

  pvalue <- 2 * pt(-abs(beta_hat / sqrt(beta_hat_var)), df = length(y) - 2)

  return(pvalue)
}

# Robust cholesky decomposition using torch
robust_chol <- function(A, tol = 1e-6, upper = FALSE) {
  tryCatch(L <- linalg_cholesky(A), error = function(e) {
    # First fallback - jittering
    jitter <- tol
    sucess <- FALSE
    while (!sucess & jitter < 1) {
      tryCatch(L <- linalg_cholesky(A + jitter * torch_eye(A$size(1))), error = function(e) {
        jitter <- jitter * 10
      })
      sucess <- TRUE
    }

    if (jitter >= 1) {
      # If jittering fails, fall back on eigen decomposition - very expensive to evaluate!
      # Eigenvalue decomposition
      eigen_result <- linalg_eigh(A)
      evals <- eigen_result[[1]]
      evecs <- eigen_result[[2]]

      evals[evals < tol] <- tol

      # Reconstruct A_star
      A_star <- torch_bmm(
        torch_bmm(evecs, torch_diag_embed(evals, dim1 = -2, dim2 = -1)),
        evecs$permute(c(1, 3, 2))
      )

      # Cholesky decomposition
      L <- linalg_cholesky(A_star)
    }
  })

  return(L)
}
