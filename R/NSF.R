#' Neural Spline Flow -- Autoregressive (NSF_AR)
#'
#' Implements the autoregressive neural spline flow from Durkan et al. (2019)
#' using rational-quadratic splines. Conforms to the flow interface expected
#' by \code{GPR_class}: \code{forward(z)} returns a list with \code{zk} and
#' \code{log_diag_j}.
#'
#' @param d integer, dimensionality of the input
#' @param K integer, number of spline bins (default 5)
#' @param B numeric, tail bound for the spline (default 3)
#' @param hidden_dim integer, hidden layer width (default 64)
#'
#' @return An \code{nn_module} implementing the NSF-AR layer.
#' @export
nsf_ar <- nn_module(
  classname = "NSF_AR",

  initialize = function(d, K = 5, B = 3, hidden_dim = 64) {
    self$d <- d
    self$K <- K
    self$B <- B
    self$n_params <- 3L * K - 1L

    # First dimension has no conditioning input, so we use a learned parameter
    self$init_param <- nn_parameter(torch_randn(self$n_params) * 0.1)

    # One network per dimension (dim i conditioned on dims 1..i-1)
    self$layers <- nn_module_list()
    for (i in seq_len(d - 1)) {
      self$layers$append(
        nn_sequential(
          nn_linear(i, hidden_dim),
          nn_tanh(),
          nn_linear(hidden_dim, hidden_dim),
          nn_tanh(),
          nn_linear(hidden_dim, self$n_params)
        )
      )
    }
  },

  forward = function(z) {
    batch_size <- z$shape[1]
    log_diag_j <- torch_zeros(batch_size, device = z$device)
    zk_list <- list()

    for (i in seq_len(self$d)) {
      if (i == 1L) {
        params <- self$init_param$unsqueeze(1)$expand(c(batch_size, -1L))
      } else {
        params <- self$layers[[i - 1L]](torch_cat(zk_list, dim = 2L))
      }

      out <- self$parse_and_transform(z[, i], params)
      zk_list[[i]] <- out$y$unsqueeze(2L)
      log_diag_j <- log_diag_j + out$ld
    }

    zk <- torch_cat(zk_list, dim = 2L)
    return(list(zk = zk, log_diag_j = log_diag_j))
  },

  # Parse network output into spline parameters and apply the RQS transform
  parse_and_transform = function(x, params) {
    splits <- torch_split(params, self$K, dim = -1L)
    W <- splits[[1]]
    H <- splits[[2]]
    D <- splits[[3]]

    W <- nnf_softmax(W, dim = -1L) * 2 * self$B
    H <- nnf_softmax(H, dim = -1L) * 2 * self$B
    D <- nnf_softplus(D)

    # Pad derivatives at boundaries to equal 1
    ones <- torch_ones(c(x$shape[1], 1L), device = x$device)
    D <- torch_cat(list(ones, D, ones), dim = -1L)

    self$rqs_forward(x, W, H, D)
  },

  # Rational-quadratic spline forward transform with linear tails.
  # Computes spline for all elements, uses torch_where to avoid in-place ops.
  rqs_forward = function(x, W, H, D) {
    B <- self$B
    inside <- (x >= -B) & (x <= B)

    # Cumulative widths and heights (knot positions)
    cumW <- torch_cumsum(W, dim = -1L) - B
    cumH <- torch_cumsum(H, dim = -1L) - B

    # Prepend -B
    cumW <- torch_cat(list(
      torch_full(c(cumW$shape[1], 1L), -B, device = x$device),
      cumW
    ), dim = -1L)
    cumH <- torch_cat(list(
      torch_full(c(cumH$shape[1], 1L), -B, device = x$device),
      cumH
    ), dim = -1L)

    # Clamp x into [-B, B] for safe spline computation on all elements.
    # torch_where at the end discards results for outside elements.
    x_safe <- torch_clamp(x, min = -B, max = B)

    # Find which bin each input falls into (searchsorted is 0-indexed)
    bin_idx <- torch_searchsorted(
      cumW[, 2:-1]$contiguous(), x_safe$unsqueeze(-1L)
    )$squeeze(-1L)
    bin_idx <- (bin_idx + 1L)$clamp(min = 1L, max = self$K)

    # Gather bin parameters
    idx <- bin_idx$unsqueeze(-1L)$to(dtype = torch_long())
    w_k <- cumW$gather(2, idx)$squeeze(-1L)
    w_k1 <- cumW$gather(2, idx + 1L)$squeeze(-1L)
    h_k <- cumH$gather(2, idx)$squeeze(-1L)
    h_k1 <- cumH$gather(2, idx + 1L)$squeeze(-1L)
    d_k <- D$gather(2, idx)$squeeze(-1L)
    d_k1 <- D$gather(2, idx + 1L)$squeeze(-1L)

    delta_k <- (h_k1 - h_k) / (w_k1 - w_k + 1e-8)
    xi <- torch_clamp((x_safe - w_k) / (w_k1 - w_k + 1e-8), min = 0, max = 1)

    # RQS formula
    num <- (h_k1 - h_k) * (delta_k * xi^2 + d_k * xi * (1 - xi))
    den <- delta_k + (d_k + d_k1 - 2 * delta_k) * xi * (1 - xi)
    y_spline <- h_k + num / (den + 1e-8)

    # Log derivative
    dnum <- delta_k^2 * (d_k1 * xi^2 + 2 * delta_k * xi * (1 - xi) + d_k * (1 - xi)^2)
    log_deriv_spline <- torch_log(torch_clamp(dnum, min = 1e-8)) -
      2 * torch_log(torch_abs(den) + 1e-8)

    # Select: spline output inside [-B, B], identity outside
    y <- torch_where(inside, y_spline, x)
    ld <- torch_where(inside, log_deriv_spline, torch_zeros_like(x))

    return(list(y = y, ld = ld))
  }
)


#' Neural Spline Flow -- Coupling Layer (NSF_CL)
#'
#' Implements the coupling-layer neural spline flow from Durkan et al. (2019)
#' using rational-quadratic splines. Conforms to the flow interface expected
#' by \code{GPR_class}: \code{forward(z)} returns a list with \code{zk} and
#' \code{log_diag_j}.
#'
#' @param d integer, dimensionality of the input
#' @param K integer, number of spline bins (default 5)
#' @param B numeric, tail bound for the spline (default 3)
#' @param hidden_dim integer, hidden layer width (default 64)
#'
#' @return An \code{nn_module} implementing the NSF-CL layer.
#' @export
nsf_cl <- nn_module(
  classname = "NSF_CL",

  initialize = function(d, K = 5, B = 3, hidden_dim = 64) {
    self$d <- d
    self$K <- K
    self$B <- B
    self$n_params <- 3L * K - 1L
    self$d_lower <- d %/% 2L
    self$d_upper <- d - self$d_lower

    # First half conditions on lower to transform upper
    self$f1 <- nn_sequential(
      nn_linear(self$d_lower, hidden_dim),
      nn_tanh(),
      nn_linear(hidden_dim, hidden_dim),
      nn_tanh(),
      nn_linear(hidden_dim, self$n_params * self$d_upper)
    )

    # Second half conditions on (transformed) upper to transform lower
    self$f2 <- nn_sequential(
      nn_linear(self$d_upper, hidden_dim),
      nn_tanh(),
      nn_linear(hidden_dim, hidden_dim),
      nn_tanh(),
      nn_linear(hidden_dim, self$n_params * self$d_lower)
    )
  },

  forward = function(z) {
    batch_size <- z$shape[1]
    log_diag_j <- torch_zeros(batch_size, device = z$device)

    lower <- z[, 1:self$d_lower]
    upper <- z[, (self$d_lower + 1):self$d]

    # Step 1: lower conditions the transform of upper
    params1 <- self$f1(lower)$reshape(c(batch_size, self$d_upper, self$n_params))
    out1 <- self$parse_and_transform_batch(upper, params1)
    upper <- out1$y
    log_diag_j <- log_diag_j + out1$ld$sum(dim = 2)

    # Step 2: (transformed) upper conditions the transform of lower
    params2 <- self$f2(upper)$reshape(c(batch_size, self$d_lower, self$n_params))
    out2 <- self$parse_and_transform_batch(lower, params2)
    lower <- out2$y
    log_diag_j <- log_diag_j + out2$ld$sum(dim = 2)

    zk <- torch_cat(list(lower, upper), dim = 2)
    return(list(zk = zk, log_diag_j = log_diag_j))
  },

  # Batched version: x is (batch, d_half), params is (batch, d_half, n_params)
  parse_and_transform_batch = function(x, params) {
    splits <- torch_split(params, self$K, dim = -1L)
    W <- splits[[1]]
    H <- splits[[2]]
    D <- splits[[3]]

    W <- nnf_softmax(W, dim = -1L) * 2 * self$B
    H <- nnf_softmax(H, dim = -1L) * 2 * self$B
    D <- nnf_softplus(D)

    # Pad boundary derivatives to 1
    ones <- torch_ones(c(x$shape[1], x$shape[2], 1L), device = x$device)
    D <- torch_cat(list(ones, D, ones), dim = -1L)

    self$rqs_forward_batch(x, W, H, D)
  },

  # Batched RQS: x is (batch, d_half), parameters are (batch, d_half, K/K+1).
  # Computes spline for all elements, uses torch_where to avoid in-place ops.
  rqs_forward_batch = function(x, W, H, D) {
    B <- self$B
    inside <- (x >= -B) & (x <= B)

    # Cumulative widths and heights
    cumW <- torch_cumsum(W, dim = -1L) - B
    cumH <- torch_cumsum(H, dim = -1L) - B

    cumW <- torch_cat(list(
      torch_full(c(cumW$shape[1], cumW$shape[2], 1L), -B, device = x$device),
      cumW
    ), dim = -1L)
    cumH <- torch_cat(list(
      torch_full(c(cumH$shape[1], cumH$shape[2], 1L), -B, device = x$device),
      cumH
    ), dim = -1L)

    # Clamp x into [-B, B] for safe spline computation on all elements
    x_safe <- torch_clamp(x, min = -B, max = B)

    # Find bins (searchsorted is 0-indexed, convert to 1-indexed for gather)
    bin_idx <- torch_searchsorted(
      cumW[, , 2:-1]$contiguous(), x_safe$unsqueeze(-1L)
    )$squeeze(-1L)
    bin_idx <- (bin_idx + 1L)$clamp(min = 1L, max = self$K)

    idx <- bin_idx$unsqueeze(-1L)$to(dtype = torch_long())
    w_k <- cumW$gather(3, idx)$squeeze(-1L)
    w_k1 <- cumW$gather(3, idx + 1L)$squeeze(-1L)
    h_k <- cumH$gather(3, idx)$squeeze(-1L)
    h_k1 <- cumH$gather(3, idx + 1L)$squeeze(-1L)
    d_k <- D$gather(3, idx)$squeeze(-1L)
    d_k1 <- D$gather(3, idx + 1L)$squeeze(-1L)

    delta_k <- (h_k1 - h_k) / (w_k1 - w_k + 1e-8)
    xi <- torch_clamp((x_safe - w_k) / (w_k1 - w_k + 1e-8), min = 0, max = 1)

    # RQS formula
    num <- (h_k1 - h_k) * (delta_k * xi^2 + d_k * xi * (1 - xi))
    den <- delta_k + (d_k + d_k1 - 2 * delta_k) * xi * (1 - xi)
    y_spline <- h_k + num / (den + 1e-8)

    # Log derivative
    dnum <- delta_k^2 * (d_k1 * xi^2 + 2 * delta_k * xi * (1 - xi) + d_k * (1 - xi)^2)
    log_deriv_spline <- torch_log(torch_clamp(dnum, min = 1e-8)) -
      2 * torch_log(torch_abs(den) + 1e-8)

    # Select: spline inside [-B, B], identity outside
    y <- torch_where(inside, y_spline, x)
    ld <- torch_where(inside, log_deriv_spline, torch_zeros_like(x))

    return(list(y = y, ld = ld))
  }
)
