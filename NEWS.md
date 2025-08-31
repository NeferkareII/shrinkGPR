## Changes in Version 1.1.1:
  - Fixed typo in prior on global shrinkage parameters

## Changes in Version 1.1:
  - Added student-t process as in Shah et al. (2014) as alternative to Gaussian process
  - Added option to estimate simulated data with multicollinearity
  - Added ability to estimate marginal effects of one or two parameters and associated plotting functions
  - Increased speed of estimation through use of JIT compiled torchscript
  - Changed default number of householder transformations in Sylvester flows to be min(5, d-1)
  - Corrected documentation for mean equation in shrinkGPR
  - Fixed bug in TG/NGG density in GPR_class
