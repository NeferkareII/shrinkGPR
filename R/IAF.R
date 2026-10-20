#' @title Masked Linear Layer (internal helper for MADE)
#' @description
#' The \code{masked_linear} function implements a linear layer whose weight is elementwise
#' multiplied by a fixed binary mask. It is the building block of the masked autoregressive
#' network used by \code{iaf}: the mask enforces the autoregressive connectivity so that each
#' output depends only on a prescribed subset of the inputs. The mask is registered as a buffer,
#' so it moves with the module across devices but is not treated as a learnable parameter.
#' @name masked_linear
#' @param in_features An integer specifying the input dimension.
#' @param out_features An integer specifying the output dimension.
#' @param mask A torch tensor of shape \code{(out_features, in_features)} giving the binary
#' connectivity mask.
#' @param zero_init A logical; if \code{TRUE} the weight is initialized to zero (used for the
#' output layer so the flow starts as a clean, near-identity rescaling). Defaults to \code{FALSE}.
#' @return An \code{nn_module} object implementing a masked linear layer.
#' @keywords internal
masked_linear <- nn_module(
  classname = "MaskedLinear",
  initialize = function(in_features, out_features, mask, zero_init = FALSE) {
    self$weight <- nn_parameter(torch_empty(out_features, in_features))
    self$bias   <- nn_parameter(torch_zeros(out_features))

    if (zero_init) {
      nn_init_zeros_(self$weight)
    } else {
      nn_init_kaiming_uniform_(self$weight, a = sqrt(5))
    }

    self$mask <- nn_buffer(mask)
  },

  forward = function(x) {
    nnf_linear(x, self$weight * self$mask, self$bias)
  }
)


#' @title Inverse Autoregressive Flow (IAF)
#' @description
#' The \code{iaf} function implements a single inverse-autoregressive-flow transformation as
#' described by Kingma et al. (2016) in "Improved Variational Inference with Inverse Autoregressive
#' Flow." A masked autoregressive network (MADE; Germain et al. 2015) maps the input to
#' per-dimension affine parameters, which are applied with the numerically stable LSTM-style
#' ("forget gate") update. This provides an invertible map that scales to high-dimensional latent
#' spaces for approximating complex posteriors in Gaussian process regression models.
#' @name iaf
#' @param d An integer specifying the latent dimensionality of the input space.
#' @param hidden_dim An optional integer specifying the hidden layer width of the masked
#' autoregressive network. Defaults to \code{max(64, 2 * d)}. For full expressivity this must be
#' at least \code{d - 1}: with a smaller width, inputs whose degree exceeds \code{hidden_dim} are
#' connected to no hidden unit and are ignored by that layer.
#' @param n_hidden An integer specifying the number of hidden layers in the masked network
#' (default 1).
#' @param forget_bias A numeric added to the gate pre-activation so the flow starts near a stable
#' rescaling (default 2). Larger values start closer to the identity (\eqn{\sigma \to 1}).
#' @param perm An optional integer vector, a permutation of \code{1:d} giving the autoregressive
#' ordering for this layer. If \code{NULL}, a random permutation is drawn at init.
#' @return An \code{nn_module} object representing the inverse autoregressive flow. The module has
#' the following key components:
#' \itemize{
#'   \item \code{forward(z)}: The forward pass computes the transformed variable \code{zk} and the
#'   log determinant of the Jacobian \code{log_diag_j}.
#'   \item Internal parameters are the masked linear layers of the autoregressive network
#'   (\code{hidden_layers} and \code{out_layer}).
#' }
#' @details
#' A masked autoregressive network produces two vectors from the input, an unconstrained shift
#' \eqn{m} and gate pre-activation \eqn{s}, each depending only on the earlier elements of \code{z}
#' under the layer's ordering. The transformation is the LSTM-style update
#' \deqn{\sigma = \mathrm{sigmoid}(s + b), \qquad z' = \sigma \odot z + (1 - \sigma) \odot m,}
#' where \eqn{b} is the forget bias. Because \eqn{m} and \eqn{s} are autoregressive with respect to
#' \code{z}, the Jacobian is triangular (under the layer's ordering) with \eqn{\sigma} on the
#' diagonal, so the log determinant is
#' \deqn{\log |det J| = \sum_{i=1}^d \log \sigma_i.}
#'
#' The autoregressive structure is enforced through the connectivity masks of Germain et al. (2015).
#' Each input, hidden, and output unit is assigned a degree; a connection is kept only when the
#' degrees satisfy the masking inequalities, with a strict inequality at the output layer so that no
#' output depends on its own input. Ordering is varied across stacked layers by assigning each layer
#' its own random permutation of the degrees (fixed at init), which generalizes the order-reversal
#' trick of the original paper; a permutation is volume-preserving and so does not enter
#' \code{log_diag_j}.
#'
#' @examples
#' if (torch::torch_is_installed()) {
#'   # Example: Initialize an inverse autoregressive flow
#'   d <- 5
#'   flow <- iaf(d)
#'
#'   # Forward pass through the flow
#'   z <- torch::torch_randn(10, d)  # Batch of 10 samples
#'   result <- flow(z)
#'
#'   print(result$zk)
#'   print(result$log_diag_j)
#' }
#' @references
#' Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., & Welling, M. (2016).
#' "Improved Variational Inference with Inverse Autoregressive Flow."
#' \emph{Advances in Neural Information Processing Systems 29 (NeurIPS 2016)}.
#'
#' Germain, M., Gregor, K., Murray, I., & Larochelle, H. (2015).
#' "MADE: Masked Autoencoder for Distribution Estimation."
#' \emph{Proceedings of the 32nd International Conference on Machine Learning (ICML 2015)}.
#' @export
iaf <- nn_module(
  classname = "IAF",
  initialize = function(d, hidden_dim = NULL, n_hidden = 1,
                        forget_bias = 2, perm = NULL) {

    if (is.null(hidden_dim)) {
      hidden_dim <- max(64L, 2L * d)
    }

    self$d <- d
    self$forget_bias <- forget_bias

    # Autoregressive ordering for this layer (a permutation of 1:d)
    if (is.null(perm)) perm <- sample.int(d)
    self$perm <- perm

    # Degrees: inputs and outputs share the ordering; hidden degrees cycle
    # through 1:(d-1) so that every representable degree appears
    deg_input  <- perm
    deg_output <- perm
    make_hidden_deg <- function(H, d) {
      if (d <= 1L) return(rep(1L, H))
      ((seq_len(H) - 1L) %% (d - 1L)) + 1L
    }
    deg_hidden <- vector("list", n_hidden)
    for (l in seq_len(n_hidden)) {
      deg_hidden[[l]] <- make_hidden_deg(hidden_dim, d)
    }

    # Input to first hidden: keep connection when hidden degree >= input degree
    m_in <- outer(deg_hidden[[1]], deg_input,
                  FUN = function(a, b) as.numeric(a >= b))
    layers <- list(masked_linear(d, hidden_dim, torch_tensor(m_in)))

    # Hidden to hidden: keep connection when later degree >= earlier degree
    if (n_hidden >= 2L) {
      for (l in 2:n_hidden) {
        m_hh <- outer(deg_hidden[[l]], deg_hidden[[l - 1L]],
                      FUN = function(a, b) as.numeric(a >= b))
        layers[[l]] <- masked_linear(hidden_dim, hidden_dim, torch_tensor(m_hh))
      }
    }
    self$hidden_layers <- nn_module_list(layers)

    # Last hidden to output (shift and gate blocks share a mask); strict
    # inequality so no output depends on its own input
    m_out_single <- outer(deg_output, deg_hidden[[n_hidden]],
                          FUN = function(o, h) as.numeric(o > h))
    m_out <- rbind(m_out_single, m_out_single)

    # Zero-init the output weights so at init m = s = 0 and z' = sigmoid(b) * z,
    # a stable starting transform
    self$out_layer <- masked_linear(hidden_dim, 2L * d,
                                    torch_tensor(m_out), zero_init = TRUE)
  },

  forward = function(z) {

    # Autoregressive network produces the shift and gate
    h <- z
    for (i in seq_len(length(self$hidden_layers))) {
      h <- torch_tanh(self$hidden_layers[[i]](h))
    }
    out <- self$out_layer(h)
    m <- out[, 1:self$d]
    s <- out[, (self$d + 1):(2L * self$d)]

    # LSTM-style update; triangular Jacobian with sigma on the diagonal
    sigma <- torch_sigmoid(s + self$forget_bias)
    zk    <- sigma * z + (1 - sigma) * m
    log_diag_j <- torch_log(sigma)$sum(dim = 2)

    return(list(zk = zk, log_diag_j = log_diag_j))
  }
)
