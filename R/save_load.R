#' Save a fitted shrinkGPR model object to disk
#'
#' \code{save_shrinkGPR} saves a fitted model object to a single \code{.zip} file,
#' preserving the trained model, optimizer state, and all metadata needed for
#' prediction and continued training via \code{cont_model}.
#'
#' @param obj an object of class \code{shrinkGPR}, \code{shrinkTPR},
#'   \code{shrinkMVGPR}, or \code{shrinkMVTPR}, as returned by the corresponding
#'   fitting function.
#' @param file a character string specifying the file path to save to.
#'
#' @return Invisibly returns the file path.
#'
#' @details
#' Internally, the model components are saved as separate files via
#' \code{\link[torch]{torch_save}} and bundled into a single \code{.zip} archive.
#' This is necessary because torch \code{nn_module} objects contain external pointers
#' that cannot be serialized within a nested list. The plain R components (loss history,
#' model internals) are saved via \code{\link{saveRDS}}.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   # Fit a model and save it
#'   sim <- simGPR()
#'   mod <- shrinkGPR(y ~ ., data = sim$data)
#'   tmp <- tempfile(fileext = ".zip")
#'   save_shrinkGPR(mod, tmp)
#'
#'   # Load it back
#'   mod_loaded <- load_shrinkGPR(tmp)
#'
#'   # Continue training
#'   mod2 <- shrinkGPR(y ~ ., data = sim$data, cont_model = mod_loaded)
#'   }
#' }
#' @author Peter Knaus \email{peter.knaus@@wu.ac.at}
#' @seealso \code{\link{load_shrinkGPR}}
#' @export
save_shrinkGPR <- function(obj, file) {

  valid_classes <- c("shrinkGPR", "shrinkTPR", "shrinkMVGPR", "shrinkMVTPR")
  if (!any(class(obj) %in% valid_classes)) {
    stop("'obj' must be of class ",
         paste(paste0("'", valid_classes, "'"), collapse = ", "), ".")
  }

  if (!is.character(file) || length(file) != 1) {
    stop("'file' must be a single character string.")
  }

  # Write components to a temporary directory
  tmp_dir <- tempfile("shrinkGPR_save_")
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  torch_save(obj$model, file.path(tmp_dir, "model.pt"))
  torch_save(obj$last_model, file.path(tmp_dir, "last_model.pt"))
  torch_save(obj$optimizer$state_dict(), file.path(tmp_dir, "optim_state.pt"))

  metadata <- list(
    loss            = obj$loss,
    loss_stor       = obj$loss_stor,
    model_internals = obj$model_internals,
    obj_class       = class(obj),
    device_type     = attr(obj, "device")$type
  )
  saveRDS(metadata, file.path(tmp_dir, "metadata.rds"))

  # Zip into a single file
  files_to_zip <- file.path(tmp_dir, c("model.pt", "last_model.pt",
                                       "optim_state.pt", "metadata.rds"))

  # Use -j flag to store without directory paths
  utils::zip(file, files = files_to_zip, flags = "-j")

  invisible(file)
}


#' Load a saved shrinkGPR model object from disk
#'
#' \code{load_shrinkGPR} restores a model object previously saved by
#' \code{\link{save_shrinkGPR}}, reconstructing the trained model, optimizer state,
#' and all metadata. The loaded object can be used directly for prediction or
#' passed as \code{cont_model} to continue training.
#'
#' @param file a character string specifying the file path to load from.
#'
#' @return An object of the same class as the one that was saved, with the same
#'   structure as returned by \code{shrinkGPR}, \code{shrinkTPR}, \code{shrinkMVGPR},
#'   or \code{shrinkMVTPR}.
#'
#' @details
#' If the model was originally trained on CUDA and CUDA is available on the current
#' machine, the model is loaded on CUDA. Otherwise, it is loaded on CPU with an
#' informative message. The optimizer state (including Adam momentum and adaptive
#' learning rate accumulators) is fully restored, so continued training via
#' \code{cont_model} picks up where the original run left off.
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   # Fit a model and save it
#'   sim <- simGPR()
#'   mod <- shrinkGPR(y ~ ., data = sim$data)
#'   tmp <- tempfile(fileext = ".zip")
#'   save_shrinkGPR(mod, tmp)
#'
#'   # Load and predict
#'   mod_loaded <- load_shrinkGPR(tmp)
#'   preds <- predict(mod_loaded, newdata = sim$data[1:10, ])
#'   }
#' }
#' @author Peter Knaus \email{peter.knaus@@wu.ac.at}
#' @seealso \code{\link{save_shrinkGPR}}
#' @export
load_shrinkGPR <- function(file) {

  if (!is.character(file) || length(file) != 1) {
    stop("'file' must be a single character string.")
  }

  if (!file.exists(file)) {
    stop("File '", file, "' does not exist.")
  }

  # Unzip to a temporary directory
  tmp_dir <- tempfile("shrinkGPR_load_")
  dir.create(tmp_dir)
  on.exit(unlink(tmp_dir, recursive = TRUE), add = TRUE)

  utils::unzip(file, exdir = tmp_dir)

  expected_files <- c("model.pt", "last_model.pt", "optim_state.pt", "metadata.rds")
  missing_files <- expected_files[!file.exists(file.path(tmp_dir, expected_files))]
  if (length(missing_files) > 0) {
    stop("Missing files in archive: ",
         paste(missing_files, collapse = ", "))
  }


  metadata    <- readRDS(file.path(tmp_dir, "metadata.rds"))

  # Resolve device
  if (metadata$device_type == "cuda" && cuda_is_available()) {
    device <- torch_device("cuda")
    # model$to(device = device)
    # last_model$to(device = device)
  } else {
    if (metadata$device_type == "cuda" && !cuda_is_available()) {
      message("Model was trained on CUDA but CUDA is not available. Loading on CPU.")
    }
    device <- torch_device("cpu")
  }

  # Load components
  model       <- torch_load(file.path(tmp_dir, "model.pt"), device = device)
  last_model  <- torch_load(file.path(tmp_dir, "last_model.pt"), device = device)
  optim_state <- torch_load(file.path(tmp_dir, "optim_state.pt"), device = device)




  # The internal $device attribute on the nn_module is a torch object whose
  # external pointer does not survive serialization. Re-set it.
  model$device <- device
  last_model$device <- device

  # Rebuild optimizer and restore state
  default_optim_params <- as.list(formals(optim_adam))
  default_optim_params$lr           <- 1e-4
  default_optim_params$weight_decay <- 1e-3
  default_optim_params$params       <- last_model$parameters
  optimizer <- do.call(optim_adam, default_optim_params)
  optimizer$load_state_dict(optim_state)

  res <- list(
    model           = model,
    loss            = metadata$loss,
    loss_stor       = metadata$loss_stor,
    last_model      = last_model,
    optimizer       = optimizer,
    model_internals = metadata$model_internals
  )

  attr(res, "class")  <- metadata$obj_class
  attr(res, "device") <- device

  return(res)
}
