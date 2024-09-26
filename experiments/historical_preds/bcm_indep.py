#!/usr/bin/env python
# coding: utf-8

# # Independently calculating the mean and variance for each RCM
# 25th September 2024

# Import libraries
import glob
import tqdm
import gpflow

import scipy as sp
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import tensorflow_probability as tfp

import cartopy.crs as ccrs
import cartopy.feature as cf

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.preprocessing import StandardScaler

from scipy.stats import boxcox
from scipy.special import inv_boxcox
from gpflow.conditionals.util import sample_mvn

# Custom libraries
#directory = '/data/hpcdata/users/kenzi22/'
directory = '/Users/kenzatazi/Documents/CDT/Code/'

import sys
sys.path.append(directory + 'guepard_repo/')
sys.path.append(directory + 'bcm4rcm/')
sys.path.append(directory)

import guepard
#############################################


rcm = 'CSIRO'
year = '1976_2006'
experiment = 'historical'

'''
rcm = os.environ['rcm']
year =  os.environ['year']
experiment =  os.environ['experiment']
'''

##############################################

# Get file name 
ref = experiment +  '_' + rcm + '_' + year
file = directory + 'bcm4rcm/data/processed/' + ref + '.csv'
print(file)

df = pd.read_csv(file, index_col=0)
df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].dt.month

# Boxcox transform
df['tp_tr'], lmbda = sp.stats.boxcox(df['tp']+0.001)
np.save( 'lambda_' + ref + '.npy', np.array(lmbda))

# Aggregate data by month
agg_df = pd.DataFrame()
group = df.groupby(['lat', 'lon', 'month'])
agg_df['mean'] = group.mean()['tp_tr']
agg_df['var'] = group.var()['tp_tr']

mean_scaler = StandardScaler()
df.loc['mean_z'] = mean_scaler.fit_transform(df[['mean']])

var_scaler = StandardScaler()
df.loc['var_z'] = var_scaler.fit_transform(df[['var']])
rcm_df = agg_df[['mean_z', 'var_z']].reset_index()

# Make aphrodite grid
aphro_arr= pd.read_csv(directory + 'bcm4rcm/data/aphro_grid.csv').values
aphro_rep = np.tile(aphro_arr, (12,1))
month_rep = np.repeat(np.arange(1, 13), len(aphro_arr)).reshape(-1,1)
pred_input = np.hstack([aphro_rep, month_rep])

# Split data
df_list = []
gb = rcm_df.groupby(['month'])
df_list.extend([gb.get_group(x).values for x in gb.groups])
arr = np.array(df_list)

# Prepare data
Xl = arr[:, :, :3]
Yl_mean = arr[:, :, 3][:, :, None]
Yl_var = arr[:, :, 4][:, :, None]

# Mean model

mean_kernel = gpflow.kernels.Matern32()
noise_var = 0.01
mean_submodels = guepard.utilities.get_gpr_submodels(zip(Xl, Yl_mean), mean_kernel, noise_variance=noise_var)
mean_rbcm = guepard.baselines.Ensemble(models=mean_submodels, 
                                        method=guepard.baselines.EnsembleMethods.RBCM, 
                                        weighting=guepard.baselines.WeightingMethods.VAR)

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(mean_rbcm.training_loss,
                        mean_rbcm.trainable_variables, options=dict(maxiter=100))
gpflow.utilities.print_summary(mean_rbcm)


# Var model
var_kernel = gpflow.kernels.Matern32()
noise_var = 0.01
var_submodels = guepard.utilities.get_gpr_submodels(zip(Xl, Yl_var), var_kernel, noise_variance=noise_var)
var_rbcm = guepard.baselines.Ensemble(models=var_submodels, 
                                        method=guepard.baselines.EnsembleMethods.RBCM, 
                                        weighting=guepard.baselines.WeightingMethods.VAR)

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(var_rbcm.training_loss,
                        var_rbcm.trainable_variables, options=dict(maxiter=100))
gpflow.utilities.print_summary(var_rbcm)

# Predictions
mean_ypreds, mean_upreds  = mean_rbcm.predict_y(pred_input)
var_ypreds, var_upreds = var_rbcm.predict_y(pred_input)

# Save results
result_df = pd.DataFrame()
result_df[['lat', 'lon', 'month']] = pred_input
result_df['mean'] = mean_ypreds
result_df['mean_uvar'] = mean_upreds
result_df['var'] = var_ypreds
result_df['var_uvar'] = var_upreds

result_df.to_csv(directory + 'bcm4rcm/data/outputs/' + experiment + '/bcm_' + ref + '.csv', index=False)


