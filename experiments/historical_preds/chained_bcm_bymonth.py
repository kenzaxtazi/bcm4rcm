#!/usr/bin/env python
# coding: utf-8

# # Latent BCMs 

month = '01'
rcm = 'CSIRO'
year = '1976_2006'
experiment = 'historical'
directory = '/Users/kenzatazi/Documents/CDT/Code/'


import gpflow
import tqdm
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
import sys
sys.path.append(directory + 'bcm4rcm/')
import guepard
from models import bcm
from utils.areal_plots import seasonal_means

# import sklearn scaler
from sklearn.preprocessing import StandardScaler


# Get file name 
ref = experiment +  '_' + rcm + '_' + year
file = directory + 'bcm4rcm/data/processed/' + ref + '.csv"
print(file)

# Load data
df = pd.read_csv(file, index_col=0)
loc_df =  df
loc_df['time'] = pd.to_datetime(loc_df['time'])
loc_df['month'] = loc_df['time'].dt.month
loc_df =  df[df['month'] == int(month)]
loc_df = loc_df.sort_values(by=['lon', 'lat'])

#p95 = np.percentile(loc_df['tp'], 95)

y_scaler = StandardScaler()
loc_df['tp_bc'], lmbda = sp.stats.boxcox(loc_df['tp']+0.001)
tp_tr = y_scaler.fit_transform(loc_df['tp_bc'].values.reshape(-1, 1))

loc_df['tp_tr'] = tp_tr
rcm_df = loc_df[['lon', 'lat', 'tp_tr']]

### Fit warped GP to aggregated data

# using groupby, crate categories along lon and lat columns, for rcm_df
rcm_df['lat_group'] = pd.cut(rcm_df['lat'], 10, labels=np.arange(10))
rcm_df['lon_group'] = pd.cut(rcm_df['lon'], 10, labels=np.arange(10))

df_list = []
gb = rcm_df.groupby(['lon_group', 'lat_group',])
df_list.extend([gb.get_group(x)[['lon', 'lat', 'tp_tr',]].values for x in gb.groups])

arr = np.array(df_list)
arr.shape

# Domains
Xl = arr[:, :, :2]
Yl = arr[:, :, 2][:, :, None]
Zl = arr[:, ::30, :2]


### Model

likelihood = gpflow.likelihoods.HeteroskedasticTFPConditional(
    distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
    scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
)

kernel = gpflow.kernels.SeparateIndependent(
    [
        gpflow.kernels.SquaredExponential(),  # This is k1, the kernel of f1
        gpflow.kernels.SquaredExponential(),  # this is k2, the kernel of f2
    ]
)

submodels = bcm.get_latent_submodels(Xl, Zl, kernel=kernel, likelihood=likelihood)
m_bcm1 = bcm.latent_Ensemble(
    models=submodels, method=guepard.baselines.EnsembleMethods.RBCM, weighting=guepard.baselines.WeightingMethods.VAR)

gpflow.utilities.set_trainable(m_bcm1.models[0].inducing_variable.inducing_variable_list[0].Z, False)
gpflow.utilities.set_trainable(m_bcm1.models[0].inducing_variable.inducing_variable_list[1].Z, False)
gpflow.utilities.set_trainable(m_bcm1.models[1].inducing_variable.inducing_variable_list[0].Z, False)
gpflow.utilities.set_trainable(m_bcm1.models[1].inducing_variable.inducing_variable_list[1].Z, False)


### Training

loss_fn = m_bcm1.training_loss(Xl, Yl)

variational_vars = m_bcm1.variational_variables
natgrad_opt = gpflow.optimizers.NaturalGradient(gamma=0.1)

adam_vars = tuple(m_bcm1.trainable_variables)
adam_opt = tf.optimizers.Adam(0.001)

# @tf.function
def optimisation_step():
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)


epochs = 200
log_freq = 20

for epoch in range(1, epochs + 1):
    optimisation_step()

    # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
    if epoch % log_freq == 0 and epoch > 0:
        print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")


### Predictions
aphro_grid = pd.read_csv('/Users/kenzatazi/Documents/CDT/Code/bcm4rcm/data/aphro_grid.csv')
aphro_arr = aphro_grid[['lat', 'lon']].values

ypred, var = m_bcm1.predict_y(aphro_arr, full_cov=False, full_output_cov=False)#np.array(Xl).reshape(-1,3))
arr= np.stack((ypred.numpy().flatten(), var.numpy().flatten()), axis=1)

df_temp= pd.DataFrame(arr, columns=['pred0', 'var0'])
df_temp['y_pred'] = y_scaler.inverse_transform(df_temp['pred0'].values)
df_temp['var'] = df_temp['var0'] * y_scaler.var_
df_temp[['lon', 'lat']] = aphro_arr
df_temp = df_temp[['y_pred', 'var']].fillna(0)

df_temp[['pred', 'var']].reset_index().as_csv('bcm_' + ref + '_' + str(month) + '.csv')
np.save( 'bcm_' + ref + '_' + str(month)+ '_lambda.npy', np.array(lmbda))



