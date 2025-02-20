# bcm4rcm

Robust Bayesian committee machines for regional climate model ensemble learning.

Tazi, K., Kim, S. W. P., Girona-Mata, M., & Turner, R. E. (2025). Refined climatologies of future precipitation over High Mountain Asia using probabilistic ensemble learning. [arXiv preprint arXiv:2501.15690](https://arxiv.org/abs/2501.15690)

## Code

### experiments
 * `bcm_preds`: scripts and visualisations for the robust Bayesian committee machine (BCM)
 * `ensemble_learning`: scripts and visualisations for the Wassertein ditance calculations and weight optimisations
 * `moe_preds`: scripts and visualisations for the mixture of experts (MoE) and equally weighted mixture model (EW) sampling
 * `validation`: scripts and visualisations for the MoE validation

### models
 * `guepard_baselines`: updated BCM code from the `guepard` library

### plots
 * `hma_map`: map of study area with with subregions

### utils
* `areal_plots`: functions to help plot the maps
* `process_data`: functions to process raw RCM data to make files over desired time periods

## Data
The CORDEX-WAS data are available through the [Earth System Federation Grid nodes](https://esgf-metagrid.cloud.dkrz.de} and the APHRODITE data through [APHRODITE's Water Resources website](http://aphrodite.st.hirosaki-u.ac.jp/download/). The data for the final results analysed in the paper are found on [Zenondo](https://doi.org/10.5281/zenodo.14837272)
