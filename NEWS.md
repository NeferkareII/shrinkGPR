## Changes in Version 2.0.1:
  - Updated README.md to be more informative, as well as include multivariate case
  - Fixed an issue with the distribution of the latent variables in shrinkMVGPR causing lack of convergence in some cases

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
