sylvester <- nn_module(
  classname = "Sylvester",
  initialize = function(d, n_householder) {
    self$d <- d
    self$n_householder <- n_householder

    self$h <- torch_tanh
  }
)

sylvester(5, 3)
