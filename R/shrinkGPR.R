#' @export
shrinkGPR <- function(formula, formula_mean, data, a = 0.5, c = 0.5, a_mean = 0.5, c_mean = 0.5, sigma2_rate = 10,
          kernel_func = kernel_se, n_layers = 10, n_latent_start = 5, max_latent = 50,
          flow_func = sylvester, flow_args, n_epochs = 1000, auto_stop = TRUE, cont_model, device,
          verbose = TRUE, optim_control) {

  # Add device attribute, set to GPU if available
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
  mf[[1L]] <- quote(stats::model.frame)
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

  # For mean equation
  if (!missing(formula_mean)) {
    mf_mean <- model.frame(formula = formula_mean, data = data, drop.unused.levels = TRUE, na.action = na.pass)
    mt_mean <- attr(x = mf_mean, which = "terms")

    # Create Matrix X with dummies and transformations
    x_mean <- model.matrix(object = mt_mean, data = mf_mean)

    # Check that there are no NAs in x_mean
    if (any(is.na(x_mean))) {
      stop("No NA values are allowed in covariates for the mean equation")
    }

    colnames(x_mean)[colnames(x_mean) == "(Intercept)"] <- "Intercept"
  } else {
    x_mean <- NULL
  }


  if (missing(cont_model)) {

    # Merge user and default flow_args
    if (missing(flow_args)) flow_args <- list()
    flow_args_merged <- list_merger(formals(flow_func), flow_args)

    # d is always handled internally
    flow_args_merged$d <- NULL

    # Create y, x and x_mean tensors
    y <- torch_tensor(y, device = device)
    x <- torch_tensor(x, device = device)

    if (!is.null(x_mean)) {
      x_mean <- torch_tensor(x_mean, device = device)
    }

    model <- GPR_class(y, x, x_mean, a = a, c = c, a_mean = a_mean, c_mean = c_mean,
                       sigma2_rate = sigma2_rate, n_layers, flow_func, flow_args_merged,
                        kernel_func = kernel_se, device)

    # Merge user and default optim_control
    if (missing(optim_control)) optim_control <- list()
    default_optim_params <- formals(optim_adam)
    default_optim_params$params <- model$parameters
    optim_control_merged <- list_merger(default_optim_params, optim_control)

    optimizer <- do.call(optim_adam, optim_control_merged)
  } else {
    model <- cont_model$last_model
    optimizer <- cont_model$optimizer
    best_model <- cont_model$best_model
    best_loss <- cont_model$best_loss
  }

  # Create progress bar if verbose is TRUE
  if (verbose) {
    pb <- progress_bar$new(total = n_epochs, format = "[:bar] :percent :eta", clear = FALSE)
  }

  # Create vector to store ELBO
  loss_stor <- rep(NA_real_, n_epochs)

  # Number of latent samples
  # Idea here is to start off with a small number of samples and
  # increase the number when there is no improvement in the ELBO for the last n_latent_increas iterations
  # This needs to be fairly aggressive, so the early stopping is not triggered before it is time
  n_latent_increase <- 25
  n_latent <- n_latent_start

  # Number of iterations to check for significant improvement
  n_check <- 100

  for (i in 1:n_epochs) {

    on.exit(return(list(model = best_model,
                        loss = best_loss,
                        loss_stor = loss_stor,
                        last_model = model,
                        optimizer = optimizer)))

    # Sample from base distribution
    z <- model$gen_batch(n_latent)

    # Forward pass through model
    zk_log_det_J <- model(z)
    zk_pos <- zk_log_det_J$zk
    log_det_J <- zk_log_det_J$log_det_J

    # Calculate loss, i.e. ELBO
    loss <- -model$elbo(zk_pos, log_det_J)
    loss_stor[i] <- loss$item()

    # Zero gradients
    optimizer$zero_grad()

    # Compute gradients, i.e. backprop
    loss$backward(retain_graph = TRUE)

    # Clip gradients
    # Seems to be beneficial, can get pretty extreme around 0
    nn_utils_clip_grad_norm_(model$parameters, max_norm = 1)

    # Update parameters
    optimizer$step()

    # Check if model is best
    if (i == 1) {
      best_model <- model
      best_loss <- loss$item()
    } else if (loss$item() < best_loss & !is.na(loss$item()) & !is.infinite(loss$item())) {
      best_model <- model
      best_loss <- loss$item()
    }

    # Increase number of latent variables by 1 if no significant improvement
    if (i %% n_latent_increase == 0 &
        i > (n_latent_increase - 1) &
        n_latent < max_latent) {
      X <- 1:n_latent_increase
      Y <- loss_stor[(i - n_latent_increase + 1):i]
      p_val <- lightweight_ols(Y, X)

      if (p_val > 0.05) {
        n_latent <- min(n_latent + 1, max_latent)
      }
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
        break
      }
    }

    # Update progress bar
    if (verbose) {
      pb$tick()
    }
  }

  # Return list of results
  res <- list(model = best_model,
              loss = best_loss,
              loss_stor = loss_stor,
              last_model = model,
              optimizer = optimizer)
  attr(res, "class") <- "shrinkGPR"

  return(res)
}
