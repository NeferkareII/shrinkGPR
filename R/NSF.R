#' @title Neural Spline Flow (Coupling Layer)
#' @description
#' The \code{nsf_cl} function implements a coupling-layer neural spline flow as described by
#' Durkan et al. (2019) in "Neural Spline Flows." Each layer applies a learned monotonic
#' rational-quadratic spline to one half of the inputs, conditioned on the other half, preceded
#' by an LU-decomposed invertible linear transformation that mixes the dimensions. This yields a
#' flexible, invertible map for approximating complex posteriors in Gaussian process regression models.
#' @name nsf_cl
#' @param d An integer specifying the latent dimensionality of the input space.
#' @param K An integer specifying the number of spline bins (default 10). Larger values give more
#' flexible splines at the cost of more parameters.
#' @param B A numeric specifying the tail bound of the spline (default 10). The spline acts on
#' \eqn{[-B, B]} and is the identity outside this interval.
#' @param hidden_dim An optional integer specifying the hidden layer width of the coupling networks.
#' Defaults to \code{max(64, 2 * d)}.
#' @return An \code{nn_module} object representing the neural spline flow coupling layer. The module
#' has the following key components:
#' \itemize{
#'   \item \code{forward(z)}: The forward pass computes the transformed variable \code{zk} and the
#'   log determinant of the Jacobian \code{log_diag_j}.
#'   \item Internal parameters include the LU factors (\code{L_raw}, \code{U_raw}, \code{log_diag_U})
#'   of the invertible linear map and the coupling networks \code{f1} and \code{f2}.
#' }
#' @details
#' Each layer first applies an invertible linear transformation parameterized through its LU
#' decomposition,
#' \deqn{W = P L U,}
#' where \code{P} is a fixed permutation, \code{L} is unit lower triangular, and \code{U} is upper
#' triangular with a learned positive diagonal. Its log determinant is
#' \deqn{\log |det W| = \sum_{i=1}^d \log U_{ii}.}
#'
#' The inputs are then split into two halves. One half is transformed elementwise by a monotonic
#' rational-quadratic spline whose bin widths, heights, and internal derivatives are produced by a
#' neural network conditioned on the other half; the roles of the two halves are then swapped. On
#' bin \eqn{k}, with \eqn{\xi = (x - x_k) / (x_{k+1} - x_k)}, bin slope
#' \eqn{s_k = (y_{k+1} - y_k) / (x_{k+1} - x_k)}, and boundary derivatives \eqn{\delta_k, \delta_{k+1}},
#' the spline is
#' \deqn{g(\xi) = y_k + \frac{(y_{k+1} - y_k)(s_k \xi^2 + \delta_k \xi (1 - \xi))}{s_k + (\delta_{k+1} + \delta_k - 2 s_k) \xi (1 - \xi)},}
#' with derivative
#' \deqn{g'(\xi) = \frac{s_k^2 (\delta_{k+1} \xi^2 + 2 s_k \xi (1 - \xi) + \delta_k (1 - \xi)^2)}{[s_k + (\delta_{k+1} + \delta_k - 2 s_k) \xi (1 - \xi)]^2}.}
#'
#' Because both the linear map and each coupling step have triangular Jacobians, the total log
#' determinant is the sum of the linear-map term and the per-dimension log spline derivatives:
#' \deqn{\log |det J| = \sum_{i=1}^d \log U_{ii} + \sum_i \log g'(\xi_i).}
#'
#' @examples
#' if (torch::torch_is_installed()) {
#'   # Example: Initialize a neural spline flow coupling layer
#'   d <- 5
#'   flow <- nsf_cl(d)
#'
#'   # Forward pass through the flow
#'   z <- torch::torch_randn(10, d)  # Batch of 10 samples
#'   result <- flow(z)
#'
#'   print(result$zk)
#'   print(result$log_diag_j)
#' }
#' @references
#' Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019).
#' "Neural Spline Flows."
#' \emph{Advances in Neural Information Processing Systems 32 (NeurIPS 2019)}.
#'
#' Kingma, D. P., & Dhariwal, P. (2018).
#' "Glow: Generative Flow with Invertible 1x1 Convolutions."
#' \emph{Advances in Neural Information Processing Systems 31 (NeurIPS 2018)}.
#' @export
nsf_cl <- nn_module(
  classname = "NSF_CL",
  initialize = function(d, K = 10, B = 10, hidden_dim = NULL) {

    if (is.null(hidden_dim)) {
      hidden_dim <- max(64L, 2L * d)
    }

    self$d <- d
    self$K <- K
    self$B <- B
    self$n_params <- 3L * K - 1L
    self$d_lower <- d %/% 2L
    self$d_upper <- d - self$d_lower

    # LU-decomposed invertible linear layer (W = P R1 R2), P a fixed permutation
    perm <- sample.int(d)
    self$P <- torch_zeros(d, d)
    for (i in seq_len(d)) {
      self$P[i, perm[i]] <- 1
    }
    self$P <- nn_buffer(self$P)

    # L: unit lower triangular; U: upper triangular with learned positive diagonal.
    # At init the diagonal is 1, so LU = I and W starts as the permutation P.
    self$L_raw <- nn_parameter(torch_zeros(d, d))
    self$U_raw <- nn_parameter(torch_zeros(d, d))
    self$log_diag_U <- nn_parameter(torch_zeros(d))

    # Coupling networks: each produces the spline parameters for one half,
    # conditioned on the other half
    self$f1 <- nn_sequential(
      nn_linear(self$d_lower, hidden_dim),
      nn_tanh(),
      nn_linear(hidden_dim, hidden_dim),
      nn_tanh(),
      nn_linear(hidden_dim, self$n_params * self$d_upper)
    )
    self$f2 <- nn_sequential(
      nn_linear(self$d_upper, hidden_dim),
      nn_tanh(),
      nn_linear(hidden_dim, hidden_dim),
      nn_tanh(),
      nn_linear(hidden_dim, self$n_params * self$d_lower)
    )
  },

  get_L = function() {
    torch_tril(self$L_raw, diagonal = -1L) +
      torch_eye(self$d, device = self$L_raw$device)
  },

  get_U = function() {
    torch_triu(self$U_raw, diagonal = 1L) +
      torch_diag(torch_exp(self$log_diag_U))
  },

  forward = function(z) {
    batch_size <- z$shape[1]
    log_diag_j <- torch_zeros(batch_size, device = z$device)

    # Invertible linear transform; log|det W| = sum(log diag U)
    W <- self$P$mm(self$get_L())$mm(self$get_U())
    z <- z$mm(W$t())
    log_diag_j <- log_diag_j + self$log_diag_U$sum()

    # Coupling: transform each half conditioned on the other, then swap
    lower <- z[, 1:self$d_lower]
    upper <- z[, (self$d_lower + 1):self$d]

    params1 <- self$f1(lower)$reshape(c(batch_size, self$d_upper, self$n_params))
    out1 <- self$parse_and_transform_batch(upper, params1)
    upper <- out1$y
    log_diag_j <- log_diag_j + out1$ld$sum(dim = 2)

    params2 <- self$f2(upper)$reshape(c(batch_size, self$d_lower, self$n_params))
    out2 <- self$parse_and_transform_batch(lower, params2)
    lower <- out2$y
    log_diag_j <- log_diag_j + out2$ld$sum(dim = 2)

    zk <- torch_cat(list(lower, upper), dim = 2)
    return(list(zk = zk, log_diag_j = log_diag_j))
  },

  # Split raw network outputs into spline widths, heights, and derivatives
  parse_and_transform_batch = function(x, params) {
    splits <- torch_split(params, self$K, dim = -1L)
    W <- splits[[1]]
    H <- splits[[2]]
    D <- splits[[3]]

    W <- nnf_softmax(W, dim = -1L) * 2 * self$B
    H <- nnf_softmax(H, dim = -1L) * 2 * self$B
    D <- nnf_softplus(D)

    # Boundary derivatives fixed to 1 for identity tails
    ones <- torch_ones(c(x$shape[1], x$shape[2], 1L), device = x$device)
    D <- torch_cat(list(ones, D, ones), dim = -1L)

    self$rqs_forward_batch(x, W, H, D)
  },

  # Monotonic rational-quadratic spline, evaluated on all elements via
  # torch_where to keep the operation batched and free of in-place writes
  rqs_forward_batch = function(x, W, H, D) {
    B <- self$B
    inside <- (x >= -B) & (x <= B)

    # Cumulative knot positions on [-B, B]
    cumW <- torch_cumsum(W, dim = -1L) - B
    cumH <- torch_cumsum(H, dim = -1L) - B
    cumW <- torch_cat(list(
      torch_full(c(cumW$shape[1], cumW$shape[2], 1L), -B, device = x$device), cumW
    ), dim = -1L)
    cumH <- torch_cat(list(
      torch_full(c(cumH$shape[1], cumH$shape[2], 1L), -B, device = x$device), cumH
    ), dim = -1L)

    x_safe <- torch_clamp(x, min = -B, max = B)

    # Locate the bin via the K-1 internal knots, then shift to a 1-based index
    n_cumW <- cumW$shape[3]
    bin_idx <- torch_searchsorted(
      cumW[, , 2:(n_cumW - 1L)]$contiguous(), x_safe$unsqueeze(-1L)
    )$squeeze(-1L)
    bin_idx <- (bin_idx + 1L)$clamp(min = 1L, max = self$K)
    idx <- bin_idx$unsqueeze(-1L)$to(dtype = torch_long())

    w_k  <- cumW$gather(3, idx)$squeeze(-1L)
    w_k1 <- cumW$gather(3, idx + 1L)$squeeze(-1L)
    h_k  <- cumH$gather(3, idx)$squeeze(-1L)
    h_k1 <- cumH$gather(3, idx + 1L)$squeeze(-1L)
    d_k  <- D$gather(3, idx)$squeeze(-1L)
    d_k1 <- D$gather(3, idx + 1L)$squeeze(-1L)

    delta_k <- (h_k1 - h_k) / (w_k1 - w_k + 1e-8)
    xi <- torch_clamp((x_safe - w_k) / (w_k1 - w_k + 1e-8), min = 0, max = 1)

    # Spline value and its log derivative
    num <- (h_k1 - h_k) * (delta_k * xi^2 + d_k * xi * (1 - xi))
    den <- delta_k + (d_k + d_k1 - 2 * delta_k) * xi * (1 - xi)
    y_spline <- h_k + num / (torch_abs(den) + 1e-8)

    dnum <- delta_k^2 * (d_k1 * xi^2 + 2 * delta_k * xi * (1 - xi) + d_k * (1 - xi)^2)
    log_deriv_spline <- torch_log(torch_clamp(dnum, min = 1e-8)) -
      2 * torch_log(torch_abs(den) + 1e-8)

    # Spline inside [-B, B], identity in the tails
    y  <- torch_where(inside, y_spline, x)
    ld <- torch_where(inside, log_deriv_spline, torch_zeros_like(x))

    return(list(y = y, ld = ld))
  }
)
