## Changes in Version 2.1.1:
  - Fixed bug in prior for shrinkTPR
  - Fixed bug in calc_pred_moments for shrinkTPR
  - Fixed bug in the implementation of cont_model
  - Fixed bug in the implementation of the U function in log_hyperu
  - Fixed a bug in gen_posterior_samples
  - Fixed 2D marginal plot mixing responses for multivariate models
  - Corrected the LKJ prior on the correlation matrix of shrinkMVGPR/shrinkMVTPR (missing Jacobian for M > 2) and removed a double diagonal adjustment in predict for shrinkMVGPR.
  - Removed a spurious Jacobian term for the diagonal stabilization of the output covariance from the ELBO of shrinkMVGPR/shrinkMVTPR, which penalized small output scales. Fitted results will differ from previous versions.
  - Fixed predict for shrinkTPR returning an nsamp x nsamp matrix when predicting at a single point
  - simMVGPR now rescales Omega so that the product of its marginal standard deviations is 1, making it directly comparable to shrinkMVGPR estimates. Omega is now also documented in the returned true values.
  - Documentation fixes: output scale is 1/tau, n_latent is the number of Monte Carlo samples, identification of the output scales in shrinkMVGPR/shrinkMVTPR, matrix-variate t is not vectorizable, Sylvester flow formula, Matérn distance jitter. Added reference to Chen, Wang & Gorban (2020).


## Changes in Version 2.1.0:
  - Added new flow architectures, specifically the Neural Spine Flow (Durkan et al., 2019) and the Inverse Autoregressive Flow (Kingma et al., 2016)
  - Updated README.md to be more informative, as well as include multivariate case
  - Fixed an issue with the distribution of the latent variables in shrinkMVGPR causing lack of convergence in some cases
  - Fixed a bug that didn't allow for covariates in the mean formula to be used in gen_marginal_samples()

## Changes in Version 2.0.0:
  - Potentially breaking change: Changed name of global shrinkage parameter from lambda to tau, as to be more consistent with references
  - Added support for multivariate GPR and TPR under shrinkage
  - Added method to save and load model objects
  - Overhauled documentation to be more informative, including model specification
  - Fixed various bugs
  - Fixed issues on CRAN with examples being run when torch is not installed 
  
## Changes in Version 1.1.3
  - Fixed issues on CRAN with examples being run when torch is not installed 

## Changes in Version 1.1.2
  - Moved to torch for R version >= 0.16.0
  - Made training loop more numerically robust by falling back on robust Cholesky decomposition
  - Fixed error in documentation for sylvester flows
  - Fixed issues on CRAN with examples being run when torch is not installed

## Changes in Version 1.1.1:
  - Fixed typo in prior on global shrinkage parameters
  - Fixed error when generating posterior samples with (optional) linear mean equation
  - Wrapped JIT compilation in torch::torch_is_installed() check to avoid errors when torch is not installed

## Changes in Version 1.1:
  - Added student-t process as in Shah et al. (2014) as alternative to Gaussian process
  - Added option to estimate simulated data with multicollinearity
  - Added ability to estimate marginal effects of one or two parameters and associated plotting functions
  - Increased speed of estimation through use of JIT compiled torchscript
  - Changed default number of householder transformations in Sylvester flows to be min(5, d-1)
  - Corrected documentation for mean equation in shrinkGPR
  - Fixed bug in TG/NGG density in GPR_class
