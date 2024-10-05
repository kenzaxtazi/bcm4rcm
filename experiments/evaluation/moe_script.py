#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import scipy as sp
import pandas as pd
import os
import CRPS.CRPS as pscore

from eval_utils import sample_moe, aggregate_to_4d_array, compute_kde_results, compute_weighted_data_points, compute_kde, log_transform, df_to_4d_array
from distributions import fit_distributions_to_4d_data, plot_fitted_distributions
from metrics import compute_crps
from eval_utils import compute_inverse_weights
from eval_utils import tile_with_neighbors_stride

def list_files_in_folder(folderpath, file_extension="",filename_identifier=""):
    files = []
    for file in os.listdir(folderpath):
        if file.endswith(file_extension):
            if filename_identifier in file:
                files.append(os.path.join(PATH, file))
                
    return files

#########

experiment = 'historical'
year = '1976_2005'

##########


# TO DO: remove hard code for dates

# ### List RCM outputs, lambdas and weights

# List RCM files
PATH = "data/bcm_outputs/" + experiment
files = list_files_in_folder(folderpath=PATH, file_extension=".csv",filename_identifier=year)

# List Lambda files
PATH = "data/bcm_outputs/" + experiment
lambda_files = list_files_in_folder(folderpath=PATH, file_extension=".npy",filename_identifier=year)

# List p95 files
PATH = "data/processed/"
p95_files = list_files_in_folder(folderpath=PATH, file_extension=".npy",filename_identifier='p95_')

# List Wass files
PATH = "data/weights"
wass_files = list_files_in_folder(folderpath=PATH, file_extension=".csv",filename_identifier='1976')


### Aggregate

# Dict of weights in 4D arrays [month, lon, lat, wass]
wass_dict = {}
for file in wass_files:
    df = pd.read_csv(file)
    wass_array = df_to_4d_array(df, variable='wass')
    wass_dict[(file.split('/')[-1].split('_1976_2005.csv')[0].split('wass_historical_')[1])] = wass_array

#wass_dict.keys()

# Dict of BCM outputs in 4D arrays [month, lon, lat, variable], where variables are: mean, var, mean_uvar, var_uvar
rcm_dict = {}
for file in files:
    bcm_dict = {}
    df = pd.read_csv(file)
    for k in ['mean','var','var_log','mean_uvar','var_log_uvar']:
        bcm_dict[k] = df_to_4d_array(df, variable=k)
    #rcm_dict[file.split('/')[-1].split('_')[2]] = bcm_dict
    rcm_dict[(file.split('/')[-1].split('_' + year + '.csv')[0].split(experiment + '_')[1])] = bcm_dict
    
#rcm_dict.keys()

# Dict of lambda values
lambda_dict = {}
for file in lambda_files:
    lambda_dict[(file.split('/')[-1].split('_' + year + '.npy')[0].split(experiment + '_')[1])] = np.load(file)

# Dict of p95 values
p95_dict = {}
for file in p95_files:
    p95_dict[(file.split('/')[-1].split('p95_')[1])] = np.load(file)
    
#lambda_dict.keys()

# Align dictionary keys
wass_dict = {key: wass_dict[key] for key in rcm_dict.keys()}
lambda_dict = {key: lambda_dict[key] for key in rcm_dict.keys()}

# ### Print dicts
"""
# Print shape of RCM dictionary items
for k,v in rcm_dict.items():
    print(k, v['mean'].shape)

# Print shape of Weight dictionary items
for k,v in wass_dict.items():
    print(k, v.shape)

# Print shape of lambda dictionary items
for k,v in lambda_dict.items():
    print(k, v.shape)
"""

# ### Stack dicts

# Stack BCM ouputs into 4d arrays
rcm_means = np.stack([sub_dict['mean'] for sub_dict in rcm_dict.values()], axis=0)
rcm_vars = np.stack([sub_dict['var'] for sub_dict in rcm_dict.values()], axis=0)

# Stack rcm_dict into a 4D array: [rcm, month, lon, lat]
distances_4d = np.stack(list(wass_dict.values()), axis=0)

# Stack lambdas
lambda_array = np.stack(list(lambda_dict.values()), axis=0)


# Stack p95s
p95_array = np.stack(list(p95_dict.values()), axis=0)


# ### Build BCM eCDFs

# Sample from BCM dists 
shape = (5, 12, 180, 80)
N = 100

lambda_array = np.tile(lambda_array[:, None, None, None, None],(1, 12, 180, 80, N))
p95_array = np.tile(p95_array[:, None, None, None, None],(1, 12, 180, 80, N))

# Validate variance
if np.any(rcm_vars < 0):
    raise ValueError("Variance array contains negative values, which is invalid for Gaussian distributions.")

# Compute standard deviation
rcm_stds = np.sqrt(rcm_vars).astype(np.float32)

# Expand dimensions for sampling axis
mean_expanded = rcm_means[..., np.newaxis]  # Shape: (5, 12, 180, 80, 1)
std_expanded = rcm_stds[..., np.newaxis]    # Shape: (5, 12, 180, 80, 1)

# Generate standard normal samples with float32
standard_normal_samples = np.random.randn(*shape, N).astype(np.float32)  # Shape: (5, 12, 180, 80, N)

# Scale and shift to obtain final samples
samples = mean_expanded + std_expanded * standard_normal_samples  # Shape: (5, 12, 180, 80, N)

print("Shape of the sampled array:", samples.shape)
print("Data type of the sampled array:", samples.dtype)


# ### Inverse Box Cox transformation

# Undo BoxCox transformation
samples_transformed = sp.special.inv_boxcox(samples, lambda_array)

#print(np.isnan(samples_transformed).sum())

# Quick fix: replace NaNs with zeros
samples_transformed[np.isnan(samples_transformed)]=0


## Scale output 
ref_p95 = np.load('data/processed/p95ref_aphro.npy')
samples_transformed2 = samples_transformed/p95_array * ref_p95 


# ### Plot arrays by RCM and month

"""
plot_figure = False
if plot_figure:

    cols = 5
    rows = 12
#     array = np.sqrt(rcm_vars[:,:,:,:])
#     array = rcm_means
#     array = distances_4d
    array = samples_transformed[:,:,:,:,1]

    fig, axes = plt.subplots(rows,cols, figsize=(20,30))
    for i in range(rows):
        for j in range(cols):
            ax = axes[i,j]

#             ax.imshow(distances_4d[j,i,:,:].transpose(), cmap='viridis');
            ax.imshow(array[j,i,:,:].transpose(), 
                      cmap='viridis',
                      origin='lower',
                      aspect='auto')

            n = np.isnan(array[j,i,:,:]).sum() / (80*180) * 100
            ax.set_title(f"rcm: {j}; month: {i}; nans: {n:.3f}%")

    plt.tight_layout()

print(np.isnan(distances_4d).sum())
"""

# Weights array --> NaNs to zeros
distances_4d[np.isnan(distances_4d)] = 0

# Create naive weights array
weights_equal = np.ones_like(distances_4d)*0.2

# Compute MoE weights
moe_weights = compute_inverse_weights(distances_4d)

moe_samples = sample_moe(samples=samples_transformed2, weights=moe_weights, num_moe_samples=100)
ew_samples = sample_moe(samples=samples_transformed, weights=weights_equal, num_moe_samples=100)

# Save samples 
np.save('data/moe_outputs/moe_'+ experiment + '_' + year + '_samples.npy', moe_samples)
np.save('data/moe_outputs/ew_'+ experiment + '_' + year + '_samples.npy', moe_samples)

# Saves means
np.save('data/moe_outputs/moe_'+ experiment + '_' + year + '_means.npy', np.mean(moe_samples, axis=-1))
np.save('data/moe_outputs/ew_'+ experiment + '_' + year + '_means.npy', np.mean(ew_samples, axis=-1))
