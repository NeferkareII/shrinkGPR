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
    while (!sucess & jitter < 1e-2) {
      Lower <- linalg_cholesky_ex(A + jitter * torch_eye(A$size(2), device = A$device))

      if (!Lower$info$any()$item()) {
        sucess <- TRUE
      } else {
        jitter <- jitter * 2
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

# Prevents values from being too close to zero
res_protector_autograd <- function(x, tol = 1e-6) {
  torch_clamp(x, min = tol)
}


# Merges user and default values of named list inputs
list_merger <- function(default, user) {

  # Check that user and sv_param are a list
  if (is.list(user) == FALSE | is.data.frame(user)){
    stop(paste0(deparse(substitute(user)), " has to be a list"))
  }

  stand_nam <- names(default)
  user_nam <- names(user)

  # Give out warning if an element of the parameter list is misnamed
  if (any(!user_nam %in% stand_nam)){
    wrong_nam <- user_nam[!user_nam %in% stand_nam]
    warning(paste0(paste(wrong_nam, collapse = ", "),
                   ifelse(length(wrong_nam) == 1, " has", " have"),
                   " been incorrectly named in ", deparse(substitute(user)), " and will be ignored"),
            immediate. = TRUE)
  }

  # Merge users' and default values and ignore all misnamed values
  missing_param <- stand_nam[!stand_nam %in% user_nam]
  user[missing_param] <- default[missing_param]
  user <- user[stand_nam]

  return(user)
}

# Small convenience function to check if something is a scalar
is.scalar <- function(x) is.atomic(x) && length(x) == 1

# Small input checkers
numeric_input_bad <- function(x) {
  if (is.scalar(x) == TRUE){
    return(is.na(x) | x <= 0 | is.numeric(x) == FALSE )
  } else {
    return(TRUE)
  }
}

numeric_input_bad_zer <- function(x) {
  if (is.scalar(x) == TRUE){
    return(is.na(x) | x < 0 | is.numeric(x) == FALSE )
  } else {
    return(TRUE)
  }
}

numeric_input_bad_ <- function(x) {
  if (is.scalar(x) == TRUE){
    return(is.na(x) | is.numeric(x) == FALSE )
  } else {
    return(TRUE)
  }
}

int_input_bad <- function(x) {
  if (is.scalar(x) == TRUE){
    if (is.numeric(x) == TRUE){
      return(is.na(x) | x < 0 | x %% 1 != 0)
    } else {
      return(TRUE)
    }
  } else {
    return(TRUE)
  }
}

bool_input_bad <- function(x){
  if (is.scalar(x) == TRUE){
    return(is.na(x) | is.logical(x) == FALSE)
  } else {
    return(TRUE)
  }
}

char_input_bad <- function(x){
  if (is.scalar(x) == TRUE){
    return(is.na(x) | is.character(x) == FALSE)
  } else {
    return(TRUE)
  }
}

lty_input_bad <- function(x){
  if (is.scalar(x) == TRUE){
    return((x %in% 0:6 | x %in% c("blank", "solid", "dashed", "dotted", "dotdash", "longdash", "twodash")) == FALSE)
  } else {
    return(TRUE)
  }
}

is_correlation_matrix <- function(R, tol = 1e-8) {
  if (!is.matrix(R)) return(FALSE)
  if (nrow(R) != ncol(R)) return(FALSE)

  # symmetry
  if (max(abs(R - t(R))) > tol) return(FALSE)

  # unit diagonal
  if (max(abs(diag(R) - 1)) > tol) return(FALSE)

  # bounds
  if (any(R < -1 - tol | R > 1 + tol)) return(FALSE)

  # positive definiteness
  !inherits(try(chol(R), silent = TRUE), "try-error")
}

rlkjcorr <- function(n, K, eta = 1) {
  stopifnot(is.numeric(K), K >= 2, K == as.integer(K))
  stopifnot(all(eta > 0))
  stopifnot(length(eta) == 1L || length(eta) == n)

  alpha <- eta + (K - 2) / 2

  r12 <- 2 * rbeta(n, alpha, alpha) - 1
  R <- array(0, dim = c(K, K, n))  # upper-triangular Cholesky factor (per draw)

  R[1, 1, ] <- 1
  R[1, 2, ] <- r12
  R[2, 2, ] <- sqrt(1 - r12^2)

  if (K > 2) {
    for (m in 2:(K - 1)) {
      alpha <- alpha - 0.5
      y <- rbeta(n, shape1 = m / 2, shape2 = alpha)  # length n

      # n independent unit vectors in R^m: columns are draws
      z <- matrix(rnorm(m * n), nrow = m, ncol = n)
      z <- z / rep(sqrt(colSums(z^2)), each = m)

      # fill column (m+1) for all draws
      R[1:m, m + 1, ] <- sweep(z, 2, sqrt(y), `*`)
      R[m + 1, m + 1, ] <- sqrt(1 - y)
    }
  }

  # correlation matrices: for each draw i, crossprod(R[,,i]) = t(R)%*%R
  out <- array(0, dim = c(K, K, n))
  for (i in 1:n) out[, , i] <- crossprod(R[, , i])

  if (n == 1L) out <- out[, , 1]
  out
}
