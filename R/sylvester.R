# This implements Sylvester normalizing flows, as described by
# van den Berg (2018) in "Sylvester Normalizing Flows for Variational Inference".
#' @export
sylvester <- nn_module(
  classname = "Sylvester",
  initialize = function(d, n_householder) {


    if (missing(n_householder)) {
      n_householder <- min(d, 10)
    }

    self$d <- d
    self$n_householder <- n_householder

    self$h <- torch_tanh

    self$diag_activation <- torch_tanh

    reg <- 1/sqrt(self$d)
    self$R1 <- nn_parameter(torch_zeros(self$d, self$d)$uniform_(-reg, reg))
    self$R2 <- nn_parameter(torch_zeros(self$d, self$d)$uniform_(-reg, reg))

    self$diag1 <- nn_parameter(torch_zeros(self$d)$uniform_(-reg, reg))
    self$diag2 <- nn_parameter(torch_zeros(self$d)$uniform_(-reg, reg))

    self$Q <- nn_parameter(torch_zeros(self$n_householder, self$d)$uniform_(-reg, reg))

    self$b <- nn_parameter(torch_zeros(self$d)$uniform_(-reg, reg))

    triu_mask <- torch_triu(torch_ones(self$d, self$d), diagonal = 1)$requires_grad_(FALSE)
    # For some reason, torch_arange goes from a to b if dtype is not specified
    # but from a to b - 1 if dtype is set to torch_long()
    diag_idx <- torch_arange(1, self$d, dtype = torch_long())$requires_grad_(FALSE)
    identity <- torch_eye(self$d, self$d)$requires_grad_(FALSE)

    self$triu_mask <- nn_buffer(triu_mask)
    self$diag_idx <- nn_buffer(diag_idx)
    self$identity <- nn_buffer(identity)

    self$register_buffer("triu_mask", triu_mask)
    self$register_buffer("diag_idx", diag_idx)
    self$register_buffer("eye", identity)

  },

  der_tanh = function(x) {
    return(1 - self$h(x)^2)
  },

  der_h = function(x) {
    return(1 - self$h(x)^2)
  },

  forward = function(zk) {

    # Bring all flow parameters into right shape
    r1 <- self$R1 * self$triu_mask
    r2 <- self$R2 * self$triu_mask

    diag1 <- self$diag_activation(self$diag1)
    diag2 <- self$diag_activation(self$diag2)

    r1$index_put_(list(self$diag_idx, self$diag_idx), diag1)
    r2$index_put_(list(self$diag_idx, self$diag_idx), diag2);

    # Orthogonalize q via Householder reflections
    norm <- torch_norm(self$Q, p = 2, dim = 2)
    v <- torch_div(self$Q$t(), norm)
    hh <- self$eye - 2 * torch_matmul(v[,1]$unsqueeze(2), v[,1]$unsqueeze(2)$t())

    for (i in 2:self$n_householder) {
      hh <- torch_matmul(hh, self$eye - 2 * torch_matmul(v[,i]$unsqueeze(2), v[,i]$unsqueeze(2)$t()))
    }

    # Compute QR1 and QR2
    rqzb <- torch_matmul(r2, torch_matmul(hh$t(), zk$t())) + self$b$unsqueeze(2)
    z <- zk + torch_matmul(hh, torch_matmul(r1, self$h(rqzb)))$t()

    # Compute log|det J|
    # Output log_det_j in shape (batch_size) instead of (batch_size,1)
    diag_j = diag1 * diag2
    diag_j = self$der_h(rqzb)$t() * diag_j
    diag_j = diag_j + 1.

    log_diag_j = diag_j$abs()$log()$sum(2)

    return(list(zk = z, log_diag_j = log_diag_j))

  }
)
