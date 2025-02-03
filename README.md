# bcm4rcm

Robust Bayesian committee machines for regional climate model ensemble learning

Tazi, K., Kim, S. W. P., Girona-Mata, M., & Turner, R. E. (2025). Refined climatologies of future precipitation over High Mountain Asia using probabilistic ensemble learning. [arXiv preprint arXiv:2501.15690](https://arxiv.org/abs/2501.15690)

experiments
 * `historical_preds`: scripts and visualisations for the Bayesian committee machine
 * `ensemble_learning`: scripts and visualisations for the Wassertein ditance calculations, weight optimisations and the mixture of experts (MoE) and equally weighted mixture model (EW) sampling
 * `future_preds`: scripts and visualisations for 
 * `validation`: scripts and visualisations for the MoE validation

models:
 * `bcm`
 * `moe`

plots:
 * `hma_map`: map of study area with with subregions

utils:
* `areal_plots`: functions to help plot the maps
* `process_data`: functions to process raw RCM data to make files over desired time periods
