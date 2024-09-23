sqdist <- function(x, l2, x_star = NULL) {
  X_l2 <- x$unsqueeze(3) * torch_sqrt(l2$t())
  sq <- torch_sum(X_l2^2, 2, keepdim = TRUE)

  if (is.null(x_star)) {

    sqdist <- (sq + sq$permute(c(2, 1, 3)))$permute(c(3, 1, 2)) -
      2 * torch_bmm(X_l2$permute(c(3, 1, 2)), X_l2$permute(c(3, 2, 1)))

  } else {

    X_star_l2 <- x_star$unsqueeze(3) * torch_sqrt(l2$t())
    sq_star <- torch_sum(X_star_l2^2, 2, keepdim = TRUE)
    sqdist <- (sq_star + sq$permute(c(2, 1, 3)))$permute(c(3, 1, 2)) -
      2 * torch_bmm(X_star_l2$permute(c(3, 1, 2)), X_l2$permute(c(3, 2, 1)))

  }
  return(sqdist)
}

kernel_se <- function(l2, lam, x, x_star = NULL) {
  sqdist_x <- sqdist(x, l2, x_star)
  K <- 1/(lam$unsqueeze(2)$unsqueeze(2)) * torch_exp(-0.5 * sqdist_x)
}

kernel_matern_12 <- function(l2, lam, x, x_star = NULL) {
  sqdist <- torch_sqrt(sqdist(x, l2, x_star) + 1e-4)
  K <- 1/(lam$unsqueeze(2)$unsqueeze(2)) * torch_exp(-sqdist)
}

kernel_matern_32 <- function(l2, lam, x, x_star = NULL) {
  sqdist <- torch_sqrt(sqdist(x, l2, x_star) + 1e-4)
  K <- 1/(lam$unsqueeze(2)$unsqueeze(2)) * (1 + sqrt(3) * sqdist) * torch_exp(-sqrt(3) * sqdist)
}

kernel_matern_52 <- function(l2, lam, x, x_star = NULL) {
  sqdist <- torch_sqrt(sqdist(x, l2, x_star) + 1e-4)
  K <- 1/(lam$unsqueeze(2)$unsqueeze(2)) * (1 + sqrt(5) * sqdist + 5/3 * sqdist^2) * torch_exp(-sqrt(5) * sqdist)
}
