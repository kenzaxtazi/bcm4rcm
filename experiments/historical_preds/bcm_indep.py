#!/usr/bin/env python
# coding: utf-8

# # Independently calculating the mean and variance for each RCM
# 25th September 2024

# Import libraries
import os
import gpflow

import scipy as sp
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

# Custom libraries
directory = '/data/hpcdata/users/kenzi22/'
#directory = '/Users/kenzatazi/Documents/CDT/Code/'

import sys
sys.path.append(directory + 'guepard_repo/')
sys.path.append(directory + 'bcm4rcm/')
sys.path.append(directory)
from models import guepard_baselines

#############################################

'''
rcm = 'CSIRO'
year = '1976_2005'
experiment = 'historical'
'''

rcm = os.environ['rcm']
year =  os.environ['year']
experiment =  os.environ['experiment']


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
np.save(directory + 'bcm4rcm/data/bcm_outputs/'+  experiment + '/lambda_' + ref + '.npy', np.array(lmbda))

# Aggregate data by month
agg_df = pd.DataFrame()
group = df.groupby(['lat', 'lon', 'month'])
agg_df['mean'] = group['tp_tr'].mean()
agg_df['var'] = group['tp_tr'].var()

mean_scaler = StandardScaler()
agg_df['mean_z'] = mean_scaler.fit_transform(agg_df['mean'].values.reshape(-1,1))

var_scaler = StandardScaler()
agg_df['var_z'] = var_scaler.fit_transform(agg_df['var'].values.reshape(-1,1))
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
mean_kernel = gpflow.kernels.Matern32(lengthscales=[1.0,1.0,1.0])
noise_var = 0.01
mean_submodels = guepard_baselines.get_gpr_submodels(zip(Xl, Yl_mean), mean_kernel, noise_variance=noise_var)
mean_rbcm = guepard_baselines.Ensemble(models=mean_submodels,
                                        method=guepard_baselines.EnsembleMethods.RBCM, 
                                        weighting=guepard_baselines.WeightingMethods.VAR)

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(mean_rbcm.training_loss,
                        mean_rbcm.trainable_variables, options=dict(maxiter=100))
gpflow.utilities.print_summary(mean_rbcm)


# Var model
var_kernel = gpflow.kernels.Matern32(lengthscales=[1,1,1])
noise_var = 0.01
var_submodels = guepard_baselines.get_gpr_submodels(zip(Xl, Yl_var), var_kernel, noise_variance=noise_var)
var_rbcm = guepard_baselines.Ensemble(models=var_submodels,
                                        method=guepard_baselines.EnsembleMethods.RBCM, 
                                        weighting=guepard_baselines.WeightingMethods.VAR)

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
result_df['mean'] = mean_scaler.inverse_transform(mean_ypreds)
result_df['mean_uvar'] = mean_upreds * mean_scaler.var_
result_df['var'] = var_scaler.inverse_transform(var_ypreds)
result_df['var_uvar'] = var_upreds * var_scaler.var_

result_df.to_csv(directory + 'bcm4rcm/data/bcm_outputs/' + experiment + '/bcm_' + ref + '.csv', index=False)



