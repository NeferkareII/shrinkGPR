# Function that returns p-value for simple linear regression
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
# Function currently only works for batched matrices
# If to be used for single matrices, do torch_unsqueeze(A, 1) before calling the function
robust_chol <- function(A, tol = 1e-6, upper = FALSE) {

  Lower <- linalg_cholesky_ex(A)

  if (Lower$info$any()$item()) {
    # First fallback - jittering
    jitter <- tol
    sucess <- FALSE
    while (!sucess & jitter < 1) {
      Lower <- linalg_cholesky_ex(A + jitter * torch_eye(A$size(2)))

      if (!Lower$info$any()$item()) {
        sucess <- TRUE
      } else {
        jitter <- jitter * 10
      }
    }
  }

  if (Lower$info$any()$item()) {
    # Second fallback - eigen decomposition
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
    Lower <- linalg_cholesky_ex(A_star)
  }

  # Give up and crawl into a hole
  if (Lower$info$any()$item()) {
    stop("Cholesky decomposition failed")
  }


  if (upper) {
    return(Lower$L$permute(1, 3, 2))
  } else {
    return(Lower$L)
  }
}

res_protector_autograd = autograd_function(
  forward = function(ctx, x) {
    result = torch_where(torch_abs(x) < 1e-10,
                         torch_sign(x) * 1e-10, x)
    return(result)
  },
  backward = function(ctx, grad_output) {
    return(grad_output)
  }
)


