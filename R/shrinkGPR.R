#' @export
shrinkGPR <- function(formula, formula_mean, data, a = 0.5, c = 0.5, a_mean = 0.5, c_mean = 0.5, sigma2_rate = 10,
                      kernel_func = kernel_se, n_layers = 10, n_latent_start = 5, auto_increase = FALSE, max_latent = 50, n_latent_increase = 25,
                      flow_func = sylvester, flow_args, n_epochs = 1000, auto_stop = TRUE, cont_model, device,
                      display_progress = TRUE, optim_control) {

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

    # Print initializing parameters message
    if (display_progress) {
      message("Initializing parameters...")
    }

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

  # Create progress bar if display_progress is TRUE
  if (display_progress) {
    pb <- progress_bar$new(total = n_epochs, format = "[:bar] :percent :eta | :message",
                           clear = FALSE, width = 100)
  }

  # Create vector to store ELBO
  loss_stor <- rep(NA_real_, n_epochs)

  # Number of latent samples
  # Idea here is to start off with a small number of samples and
  # increase the number when there is no improvement in the ELBO for the last n_latent_increas iterations
  # This needs to be fairly aggressive, so the early stopping is not triggered before it is time
  n_latent <- n_latent_start

  # Number of iterations to check for significant improvement
  n_check <- 100

  # Initialize a variable to track whether the loop exited normally or due to interruption
  stop_reason <- "max_iterations"
  runtime <- system.time({
    tryCatch({
      for (i in 1:n_epochs) {

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
        if (auto_increase) {
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

          # Show number of latent variables if auto increase is on
          if (auto_increase) {
            latent_message <- paste0(", Curr. #latent: ", n_latent, ", ")
          } else {
            latent_message <- ""
          }


          # If less than 50 iterations, don't show avg loss
          if (i >= 50) {

            # Recalculate average loss every 10 iterations
            if (i %% 10 == 0) {
              avg_loss <- mean(loss_stor[(i - 49):i])
            }

            curr_message <- paste0(latent_message,
                                   avg_loss_msg,
                                   sprintf(paste0("%-", avg_loss_width, ".2f"), avg_loss))
          } else {
            curr_message <- paste0(latent_message,
                                   format("", width = nchar(avg_loss_msg) + avg_loss_width))
          }
          pb$tick(tokens = list(message = curr_message))
        }
      }
    }, interrupt = function(ex) {
      stop_reason <<- "interrupted"
      pb$terminate()
      message("\nTraining interrupted at iteration ", i, ". Returning model trained so far.")
    }, error = function(ex) {
      stop_reason <<- "error"
      pb$terminate()
      message("\nError occurred at iteration ", i, ". Returning model trained so far.")
    }, warning = function(w) {
      if (grepl("batched routines are designed for small sizes", w$message)) {
        invokeRestart("muffleWarning")  # Suppress this specific warning
      } else {
        warning(w)  # Let other warnings pass through
      }
    })
  })


  # Print messages based on how the loop ended
  if (display_progress) {
    message(paste0("Timing (elapsed): ", runtime["elapsed"], " seconds."))
    message(paste0(round( i/ runtime[3]), " iterations per second."))

    if (stop_reason == "auto_stop" & i < n_epochs) {
      message("Auto stop triggered, iteration ", i)
    } else if (stop_reason == "max_iterations") {
      message("Max iterations reached, stopping at iteration ", i)
      message("Check if convergence is reached by looking at the loss_stor attribute of the returned object")
    }
  }

  model_internals <- list(
    terms = mt,
    xlevels = .getXlevels(mt, mf),
    data = data,
    d_cov = x$shape[2]
  )

  if (!is.null(x_mean)) {
    model_internals$terms_mean <- mt_mean
    model_internals$xlevels_mean <- .getXlevels(mt_mean, mf_mean)
    model_internals$x_mean <- TRUE
    model_internals$d_mean <- x_mean$shape[2]
  } else {
    model_internals$x_mean <- FALSE
  }

  # Return list of results
  res <- list(model = best_model,
              loss = best_loss,
              loss_stor = loss_stor,
              last_model = model,
              optimizer = optimizer,
              model_internals = model_internals)

  attr(res, "class") <- "shrinkGPR"
  attr(res, "device") <- device

  return(res)
}
