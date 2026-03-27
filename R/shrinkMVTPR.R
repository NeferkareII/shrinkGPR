#' Multivariate Student-t Process Regression with Shrinkage and Normalizing Flows
#'
#' Fits a multivariate Student-t process regression (MVTPR) model to an \eqn{N \times M} response matrix \eqn{Y}. The joint
#' distribution is matrix-variate Student-t, \eqn{Y \sim \mathcal{MT}(\nu,\, 0,\, K + \sigma^2 I,\, \Omega)}, where \eqn{K} is
#' the GP kernel matrix with triple-gamma shrinkage priors on the inverse length-scales, \eqn{\Omega} is the \eqn{M \times M}
#' output covariance, and \eqn{\nu} is the degrees of freedom parameter. Compared to \code{\link{shrinkMVGPR}}, the heavier tails
#' provide greater robustness to outliers. The joint posterior is approximated by normalizing flows trained to maximize the ELBO.
#'
#' @param formula object of class "formula": a symbolic representation of the model for the covariance equation, as in \code{\link{lm}}.
#' The response variable and covariates are specified here. Specifically, the response is created by binding the \eqn{M} response variables together with 
#' \code{cbind()} on the left-hand side of the formula, e.g., \code{cbind(y1, y2) ~ x}.
#' @param data \emph{optional} data frame containing the response variable and the covariates. If not found in \code{data},
#' the variables are taken from \code{environment(formula)}. No \code{NA}s are allowed in the response variable or covariates.
#' @param a positive real number controlling the behavior at the origin of the shrinkage prior for the covariance structure. The default is 0.5.
#' @param c positive real number controlling the tail behavior of the shrinkage prior for the covariance structure. The default is 0.5.
#' @param eta positive real number controlling the concentration of the LKJ prior on the correlation matrix of the output covariance.
#' Higher values push the prior towards the identity matrix. The default is 4.
#' @param a_Om positive real number controlling the behavior at the origin of the shrinkage prior for the output covariance scale parameters. The default is 0.5.
#' @param c_Om positive real number controlling the tail behavior of the shrinkage prior for the output covariance scale parameters. The default is 0.5.
#' @param sigma2_rate positive real number controlling the prior rate parameter for the residual variance. The default is 10.
#' @param nu_alpha positive real number controlling the shape parameter of the gamma prior for the degrees of freedom of the
#' matrix-t process. The default is 0.5.
#' @param nu_beta positive real number controlling the rate parameter of the shifted gamma prior for the degrees of freedom of the
#' matrix-t process. The default is 2.
#' @param kernel_func function specifying the covariance kernel. The default is \code{\link{kernel_se}}, a squared exponential kernel.
#' For guidance on how to provide a custom kernel function, see Details.
#' @param n_layers positive integer specifying the number of flow layers in the normalizing flow. The default is 10.
#' @param n_latent positive integer specifying the dimensionality of the latent space for the normalizing flow. The default is 10.
#' @param flow_func function specifying the normalizing flow transformation. The default is \code{\link{sylvester}}.
#' For guidance on how to provide a custom flow function, see Details.
#' @param flow_args \emph{optional} named list containing arguments for the flow function. If not provided, default arguments are used.
#' For guidance on how to provide a custom flow function, see Details.
#' @param n_epochs positive integer specifying the number of training epochs. The default is 1000.
#' @param auto_stop logical value indicating whether to enable early stopping based on convergence. The default is \code{TRUE}.
#' @param cont_model \emph{optional} object returned from a previous \code{shrinkMVTPR} call, enabling continuation of training from the saved state.
#' @param device \emph{optional} device to run the model on, e.g., \code{torch_device("cuda")} for GPU or \code{torch_device("cpu")} for CPU.
#' Defaults to GPU if available; otherwise, CPU.
#' @param display_progress logical value indicating whether to display progress bars and messages during training. The default is \code{TRUE}.
#' @param optim_control \emph{optional} named list containing optimizer parameters. If not provided, default settings are used.
#'
#' @return A list object of classes \code{shrinkMVGPR} and \code{shrinkMVTPR}, containing:
#' \item{\code{model}}{The best-performing trained model.}
#' \item{\code{loss}}{The best loss value (ELBO) achieved during training.}
#' \item{\code{loss_stor}}{A numeric vector storing the ELBO values at each training iteration.}
#' \item{\code{last_model}}{The model state at the final iteration.}
#' \item{\code{optimizer}}{The optimizer object used during training.}
#' \item{\code{model_internals}}{Internal objects required for predictions and further training, such as model matrices and formulas.}
#'
#' @details
#' \strong{Model Specification}
#'
#' Given \eqn{N} observations with \eqn{d}-dimensional covariates and \eqn{M} response variables, the response matrix
#' \eqn{Y \in \mathbb{R}^{N \times M}} follows a matrix-variate Student-t distribution:
#' \deqn{Y \sim \mathcal{MT}_{N,M}(\nu,\; 0,\; K(\theta, \tau) + \sigma^2 I_N,\; \Omega),}
#' which is equivalent to
#' \deqn{\mathrm{vec}(Y) \sim t_{NM}\!\left(\nu,\; \mathbf{0},\; \Omega \otimes (K + \sigma^2 I_N)\right).}
#' Here \eqn{K_{ij} = k(x_i, x_j;\, \theta, \tau)} is the kernel matrix and \eqn{\Omega} is the \eqn{M \times M}
#' between-response covariance. The output covariance is parameterized as \eqn{\Omega = S D S}, where
#' \eqn{D} is a correlation matrix and \eqn{S = \mathrm{diag}(s_1, \ldots, s_M)} contains the marginal standard deviations.
#' The product of the diagonal elements of \eqn{S} is constrained to equal 1 to ensure identifiability.
#' The default squared exponential kernel is
#' \deqn{k(x, x';\, \theta, \tau) = \frac{1}{\tau} \exp\!\left(-\frac{1}{2} \sum_{j=1}^d \theta_j (x_j - x'_j)^2\right),}
#' where \eqn{\theta_j \ge 0} are inverse squared length-scales and \eqn{\tau > 0} is the output scale. 
#' Users can specify custom kernels by following the guidelines below, or use one of the other provided kernel functions in 
#' \code{\link{kernel_functions}}.
#' 
#' \strong{Priors}
#'
#' \deqn{\theta_j \mid \tau \sim \mathrm{TG}(a, c, \tau), \quad j = 1, \ldots, d,}
#' \deqn{\tau \sim F(2c, 2a),}
#' \deqn{\sigma^2 \sim \mathrm{Exp}(\sigma^2_\mathrm{rate}),}
#' \deqn{D \sim \mathrm{LKJ}(\eta),}
#' \deqn{s_m \mid \tau_\Omega \sim \mathrm{TG}(a_\Omega, c_\Omega, \tau_\Omega), \quad m = 1, \ldots, M,}
#' \deqn{\tau_\Omega \sim F(2c_\Omega, 2a_\Omega),}
#' \deqn{\nu - 2 \sim \mathrm{Gamma}(\nu_\alpha, \nu_\beta).}
#' The shift by 2 ensures \eqn{\nu > 2} so that the process covariance is finite.
#'
#' \strong{Inference}
#'
#' The posterior is approximated by a normalizing flow \eqn{q_\phi} trained to maximize the ELBO.
#' \code{auto_stop} triggers early stopping when the ELBO shows no significant improvement over the last 100 iterations.
#'
#' \strong{Custom Kernel Functions}
#'
#' Users can define custom kernel functions by passing them to the \code{kernel_func} argument.
#' A valid kernel function must follow the same structure as \code{\link{kernel_se}}. The function must:
#'
#' \enumerate{
#'   \item Accept arguments \code{thetas} (\code{n_latent x d}), \code{tau} (length \code{n_latent}),
#'     \code{x} (\code{N x d}), and optionally \code{x_star} (\code{N_new x d}).
#'   \item Return a \code{torch_tensor} of dimensions \code{n_latent x N x N} (if \code{x_star = NULL})
#'     or \code{n_latent x N_new x N} (if \code{x_star} is provided).
#'   \item Produce a valid positive semi-definite covariance matrix using \code{torch} tensor operations.
#' }
#'
#' See \code{\link{kernel_functions}} for documented examples.
#'
#' \strong{Custom Flow Functions}
#'
#' Users can define custom flow functions by implementing an \code{nn_module} in \code{torch}.
#' The module must have a \code{forward} method that accepts a tensor \code{z} of shape \code{n_latent x D}
#' and returns a list with:
#' \itemize{
#'   \item \code{zk}: the transformed samples, shape \code{n_latent x D}.
#'   \item \code{log_diag_j}: log-absolute-determinant of the Jacobian, shape \code{n_latent}.
#' }
#'
#' See \code{\link{sylvester}} for a documented example.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   # Simulate multivariate data
#'   torch::torch_manual_seed(123)
#'   sim <- simMVGPR(N = 100, M = 2, d = 2)
#'
#'   # Fit MVTPR model
#'   res <- shrinkMVTPR(cbind(y.1, y.2) ~ x.1 + x.2, data = sim$data)
#'
#'   # Check convergence
#'   plot(res$loss_stor, type = "l", main = "Loss Over Iterations")
#'
#'   # Check posterior of length-scale parameters
#'   samps <- gen_posterior_samples(res, nsamp = 1000)
#'   boxplot(samps$thetas)
#'
#'   # Predict at new covariate values
#'   newdata <- data.frame(x.1 = runif(10), x.2 = runif(10))
#'   y_new <- predict(res, newdata = newdata, nsamp = 500)
#'   # y_new is an array of shape nsamp x N_new x M
#'   }
#' }
#' @export
#' @author Peter Knaus \email{peter.knaus@@wu.ac.at}
shrinkMVTPR <- function(formula,
                        data,
                        a = 0.5,
                        c = 0.5,
                        eta = 4,
                        a_Om = 0.5,
                        c_Om = 0.5,
                        sigma2_rate = 10,
                        nu_alpha = 0.5,
                        nu_beta = 2,
                        kernel_func = kernel_se,
                        n_layers = 10,
                        n_latent = 10,
                        flow_func = sylvester,
                        flow_args,
                        n_epochs = 1000,
                        auto_stop = TRUE,
                        cont_model,
                        device,
                        display_progress = TRUE,
                        optim_control) {

  # Input checking ----------------------------------------------------------

  # Check if formula is valid
  if (!inherits(formula, "formula")) {
    stop("The argument 'formula' must be of class 'formula'.")
  }

  # Check if data is provided and is a data frame
  if (!missing(data) && !is.data.frame(data)) {
    stop("The argument 'data' must be a data frame.")
  }

  # Check that numeric inputs are positive scalars
  to_check_numeric <- list(
    a = a,
    c = c,
    eta = eta,
    sigma2_rate = sigma2_rate
  )

  bad_numeric <- sapply(to_check_numeric, numeric_input_bad)
  if (any(bad_numeric)) {
    bad_names <- names(to_check_numeric)[bad_numeric]
    stop(paste0(paste(bad_names, collapse = ", "),
                ifelse(length(bad_names) == 1, " must", " must all"),
                " be positive numeric scalars."))
  }

  # Check that integer inputs are positive integers
  to_check_int <- list(
    n_layers = n_layers,
    n_latent = n_latent,
    n_epochs = n_epochs
  )

  bad_int <- sapply(to_check_int, int_input_bad)
  if (any(bad_int)) {
    bad_names <- names(to_check_int)[bad_int]
    stop(paste0(paste(bad_names, collapse = ", "),
                ifelse(length(bad_names) == 1, " must", " must all"),
                " be positive integers."))
  }

  # Check flow function and arguments
  if (!is.function(flow_func)) {
    stop("The argument 'flow_func' must be a valid function.")
  }

  if (!missing(flow_args) && !is.list(flow_args)) {
    stop("The argument 'flow_args', if provided, must be a named list.")
  }

  # Check kernel function
  if (!is.function(kernel_func)) {
    stop("The argument 'kernel_func' must be a valid function.")
  }

  # Check auto_stop is logical
  if (!is.logical(auto_stop) || length(auto_stop) != 1) {
    stop("The argument 'auto_stop' must be a single logical value.")
  }

  # Check display_progress is logical
  if (!is.logical(display_progress) || length(display_progress) != 1) {
    stop("The argument 'display_progress' must be a single logical value.")
  }

  # Check continuation model (if provided)
  if (!missing(cont_model) && !is.list(cont_model)) {
    stop("The argument 'cont_model', if provided, must be a list returned by a previous 'shrinkMVGPR' call.")
  }

  # Check device
  if (!missing(device) && !inherits(device, "torch_device")) {
    stop("The argument 'device', if provided, must be a valid 'torch_device' object.")
  }

  # Check optimizer control parameters
  if (!missing(optim_control) && !is.list(optim_control)) {
    stop("The argument 'optim_control', if provided, must be a named list.")
  }

  if (!missing(device)) {
    if (!inherits(device, "torch_device")) {
      stop("The argument 'device', if provided, must be a valid 'torch_device' object.")
    }
  }

  if (!missing(cont_model)) {
    if (!inherits(cont_model, "shrinkMVTPR")) {
      stop("The argument 'cont_model', if provided, must be a list returned by a previous 'shrinkMVTPR' call.")
    }
  }

  # Add default device if not provided -------------------------------------
  if (missing(device)) {
    if (cuda_is_available()) {
      device <- torch_device("cuda")
    } else {
      device <- torch_device("cpu")
    }
  }

  # Formula interface -------------------------------------------------------
  # For main covar equation
  mf <- match.call(expand.dots = FALSE)
  m <- match(x = c("formula", "data"), table = names(mf), nomatch = 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- TRUE
  mf$na.action <- na.pass
  mf[[1L]] <- quote(model.frame)
  mf <- eval(expr = mf, envir = parent.frame())

  # Create Vector y
  y <- model.response(mf, "numeric")

  # Modify the formula to exclude intercept
  mt <- attr(x = mf, which = "terms")
  attr(mt, "intercept") <- 0

  # Create Matrix X with dummies and transformations
  x <- model.matrix(object = mt, data = mf)

  # Check that there are no NAs in y and x
  if (any(is.na(y))) {
    stop("No NA values are allowed in response variable")
  }

  if (any(is.na(x))){
    stop("No NA values are allowed in covariates")
  }

  if (missing(cont_model)) {

    # Print initializing parameters message
    if (display_progress) {
      message("Initializing parameters...", appendLF = FALSE)
    }

    # Merge user and default flow_args
    if (missing(flow_args)) flow_args <- list()
    flow_args_merged <- list_merger(formals(flow_func), flow_args)

    # d is always handled internally
    flow_args_merged$d <- NULL

    # Create y, x tensors
    y <- torch_tensor(y, device = device)
    x <- torch_tensor(x, device = device)


    model <- MVTPR_class(y, x,  a = a, c = c, eta = eta, a_Om = a_Om, c_Om = c_Om,
                         nu_alpha = nu_alpha, nu_beta = nu_beta,
                         sigma2_rate = sigma2_rate, n_layers, flow_func, flow_args_merged,
                         kernel_func = kernel_func, device)

    # Merge user and default optim_control
    if (missing(optim_control)) optim_control <- list()
    default_optim_params <- formals(optim_adam)
    default_optim_params$lr <- 1e-4
    default_optim_params$weight_decay <- 1e-3
    default_optim_params$params <- model$parameters
    optim_control_merged <- list_merger(default_optim_params, optim_control)

    optimizer <- do.call(optim_adam, optim_control_merged)

    if (display_progress) {
      message("Done!")
    }

  } else {
    model <- cont_model$last_model
    optimizer <- cont_model$optimizer
    best_model <- cont_model$best_model
    best_loss <- cont_model$best_loss
  }

  # Create progress bar if display_progress is TRUE
  if (display_progress) {
    pb <- progress_bar$new(total = n_epochs, format = "[:bar] :percent :eta | :message",
                           clear = FALSE, width = 100)
  }

  # Create vector to store ELBO
  loss_stor <- rep(NA_real_, n_epochs)


  # Number of iterations to check for significant improvement
  n_check <- 100

  # Rolling window parameters for adaptive skip-step rule
  # Rolling window size
  w <- 50L
  # Multiplier for MAD to set cap
  k_mad <- 10
  # safety floor so cap doesn't get too small early
  cap_min <- 1e4

  # Initialize a variable to track whether the loop exited normally or due to interruption
  stop_reason <- "max_iterations"
  runtime <- system.time({
    # tryCatch({
      for (i in 1:n_epochs) {

        # Sample from base distribution
        z <- model$gen_batch(n_latent)

        # Forward pass through model
        zk_log_det_J <- model(z)
        zk_pos <- zk_log_det_J$zk
        log_det_J <- zk_log_det_J$log_det_J


        # Calculate loss, i.e. ELBO
        # suppressWarnings because torchscript does not yet support torch.linalg.cholesky
        loss <- suppressMessages(-model$elbo(zk_pos, log_det_J))

        # Zero gradients
        optimizer$zero_grad()

        # Compute gradients, i.e. backprop
        loss$backward()

        # Clip gradients to avoid exploding gradients
        nn_utils_clip_grad_norm_(model$parameters, max_norm = 0.5)

        # Update parameters
        optimizer$step()

        # Store loss value
        loss_stor[i] <- loss$item()

        # Check if model is best
        if (i == 1) {
          best_model <- model$clone(deep = TRUE)
          best_loss <- loss$item()
        } else if (loss$item() < best_loss & !is.na(loss$item()) & !is.infinite(loss$item())) {
          best_model <- model$clone(deep = TRUE)
          best_loss <- loss$item()
        }


        # Auto stop if no improvement in n_check iterations
        if (auto_stop &
            i %% n_check == 0 &
            i > (n_check - 1)) {
          X <- 1:n_check
          Y <- loss_stor[(i - n_check + 1):i]
          p_val <- lightweight_ols(Y, X)

          # Slightly more lenient here, false positives are not as bad as false negatives
          if (p_val > 0.2) {
            stop_reason <- "auto_stop"
            break
          }
        }

        # Update progress bar
        if (display_progress) {

          # Prepare message, this way width can be set
          avg_loss_msg <- "Avg. loss last 50 iter.: "
          avg_loss_width <- 7


          # If less than 50 iterations, don't show avg loss
          if (i >= 50) {

            # Recalculate average loss every 10 iterations
            if (i %% 10 == 0) {
              avg_loss <- mean(loss_stor[(i - 49):i])
            }

            curr_message <- paste0(avg_loss_msg,
                                   sprintf(paste0("%-", avg_loss_width, ".2f"), avg_loss))
          } else {
            curr_message <- format("", width = nchar(avg_loss_msg) + avg_loss_width)
          }
          pb$tick(tokens = list(message = curr_message))
        }
      }
    # }, interrupt = function(ex) {
    #   stop_reason <<- "interrupted"
    #   if (display_progress) {
    #     pb$terminate()
    #   }
    #   message("\nTraining interrupted at iteration ", i, ". Returning model trained so far.")
    # }, error = function(ex) {
    #   stop_reason <<- "error"
    #   if (display_progress) {
    #     pb$terminate()
    #   }
    #   message("\nError occurred at iteration ", i, ". Returning model trained so far.")
    # })
  })


  # Print messages based on how the loop ended
  if (display_progress) {

    if (stop_reason %in% c("auto_stop")) {
      pb$terminate()
    }

    message(paste0("Timing (elapsed): ", round(runtime["elapsed"], 2), " seconds."))
    message(paste0(round( i/ runtime[3]), " iterations per second."))

    if (stop_reason == "auto_stop" & i < n_epochs) {
      message("Auto stop triggered, iteration ", i)
    } else if (stop_reason == "max_iterations") {
      message("Max iterations reached, stopping at iteration ", i)
      message("Check if convergence is reached by looking at the loss_stor attribute of the returned object")
    }
  }

  if (missing(cont_model)) {
    model_internals <- list(
      terms = mt,
      xlevels = .getXlevels(mt, mf),
      data = data,
      d_cov = x$shape[2],
      M = y$shape[2]
    )
  } else {
    model_internals <- cont_model$model_internals
  }


  # Return list of results
  res <- list(model = best_model,
              loss = best_loss,
              loss_stor = loss_stor,
              last_model = model,
              optimizer = optimizer,
              model_internals = model_internals)

  attr(res, "class") <- c("shrinkMVGPR", "shrinkMVTPR")
  attr(res, "device") <- device

  return(res)
}
