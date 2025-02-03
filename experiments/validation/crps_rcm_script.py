#!/usr/bin/env python
# coding: utf-8

# # Independently calculating the mean and variance for each RCM
# 25th September 2024

# Import libraries
import os
import glob
import sys
import tqdm
import scipy as sp
import numpy as np
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Custom libraries
directory = '/data/hpcdata/users/kenzi22/'
#directory = '/Users/kenzatazi/Documents/CDT/Code/'

sys.path.append(directory + 'bcm4rcm/')
sys.path.append(directory)
from load import aphrodite
from experiments.evaluation.metrics import compute_crps

#############################################

target_year = '1981_2005'
target_experiment = 'historical'
ref_year = '1951_1980'
weighting = 'alpha'

'''
target_year = os.environ['target_year']
ref_year =  os.environ['ref_year']
target_experiment =  os.environ['target_experiment']
weighting = os.environ['weighting']
print(weighting)
'''
##############################################


### Identify data

# Get BCM filenames 
bcm_path = directory + 'bcm4rcm/data/bcm_outputs/' + target_experiment + '/bcm_' + target_experiment + '_*' + target_year + '.csv'
print(bcm_path)
bcm_files = sorted(glob.glob(bcm_path))
bcm_names = [n.split('l_')[1].split('_1')[0] for n in bcm_files]

lambda_path = directory + 'bcm4rcm/data/bcm_outputs/' + target_experiment + '/lambda_' + target_experiment + '_*' + target_year + '.npy'
print(lambda_path)
lambda_files = sorted(glob.glob(lambda_path))

# Get weights
if weighting == 'alpha':
    weight_path = directory + 'bcm4rcm/data/weights/250107_weights_beta_no_p95_5x5.npy'
    print(weight_path)
    alpha_arr = np.load(weight_path)

if weighting == 'alpha_p95':
    weight_path = directory + 'bcm4rcm/data/weights/250105_weights.npy'
    print(weight_path)
    alpha_arr = np.load(weight_path)

if weighting == 'alpha_p95_scaling':
    weight_path = directory + 'bcm4rcm/data/weights/250105_weights.npy'
    print(weight_path)
    alpha_arr = np.load(weight_path)
    aphro_p95 = np.load('/data/hpcdata/users/kenzi22/bcm4rcm/data/weights/archive/p95/p95_aphro_1951_1980.npy')
    p95_path = directory + 'bcm4rcm/data/weights/archive/p95/p95_historical_*' + ref_year + '.npy'
    print(p95_path)
    p95_files = sorted(glob.glob(p95_path))

# Get Aphrodite
years= ref_year.split('_')
aphro_ds = aphrodite.collect_APHRO('hma', minyear=years[0], maxyear=years[1])
aphro_ds2 = aphro_ds.sel(lon=slice(60,105), lat=slice(20,40))
aphro_ds3 = aphro_ds2.interpolate_na(dim='lat', method='linear')
aphro_df = aphro_ds3.to_dataframe().reset_index()#.dropna()
aphro_df['month'] = aphro_df['time'].dt.month
aphro_df.sort_values(['month', 'lon', 'lat'], inplace=True)
aphro_arr = aphro_df.tp.values.reshape(12, 180, 80, 30)


### Load data and prepare data
np.random.seed(42)

ref_df = pd.read_csv(bcm_files[0], index_col=0).reset_index()
ref_df.sort_values(['month', 'lon', 'lat'], inplace=True)
moe_df = ref_df[['month', 'lon', 'lat']]
moe_dict = dict(zip(bcm_names, [np.zeros(len(moe_df)) for i in range(len(bcm_names))]))
moe_df[bcm_names] = pd.DataFrame(moe_dict)

# Loop through each grid point

moe_sample_list = []
ew_sample_list = []

# Loop over RCMs
for j in tqdm.tqdm(range(len(bcm_files))):
    bcm_df = pd.read_csv(bcm_files[j], index_col=0).reset_index()
    bcm_df.sort_values(['month', 'lon', 'lat'], inplace=True)
    lmbda = np.load(lambda_files[j])
    
    # Generate sample from each BCM
    mu = bcm_df['mean'].values 
    std = np.sqrt(bcm_df['var'].values)

    rcm_samples = np.array([np.random.normal(m, s, 100) for m, s in zip(mu, std)])
    rcm_samp = np.nan_to_num(sp.special.inv_boxcox(rcm_samples, lmbda), 0)
    rcm_arr = rcm_samp.reshape(12, 180, 80, 100)
    rcm_crps = compute_crps(aphro_arr, rcm_arr)
    moe_df.loc[:, bcm_names[j]] = rcm_crps.flatten()

    # Generate MoE samples
    weights = alpha_arr[j].flatten()
    weights = np.nan_to_num(weights, 0)
    moe_samples = [np.random.normal(m, s, int(np.ceil(100*w))) for m, s, w in zip(mu, std, weights)]
    moe_samp = [np.nan_to_num(sp.special.inv_boxcox(ms, lmbda), 0) for ms in moe_samples]
    if weighting == 'alpha_p95_scaling':
        p95 = np.load(p95_files[j])
        moe_samp = [ms/p95 for ms in moe_samp]
    moe_sample_list.append(moe_samp)

    # Generate EW samples
    ew_samples = [np.random.normal(m, s, int(np.ceil(100/13))) for m, s in zip(mu, std)]
    ew_samp = [np.nan_to_num(sp.special.inv_boxcox(ms, lmbda), 0) for ms in ew_samples]
    ew_sample_list.append(ew_samp)

moe_list = [np.concatenate([moe_sample_list[j][i] for j in range(13)])[:100] for i in range(12*180*80)]
ew_list = [np.concatenate([ew_sample_list[j][i] for j in range(13)])[:100] for i in range(12*180*80)]

for i in range(len(moe_list)):
    if len(moe_list[i]) == 0:
        moe_list[i] = np.ones(100) * np.nan

for i in range(len(moe_list)):
    if len(ew_list[i]) == 0:
        ew_list[i] = np.ones(100) * np.nan

moe_arr = np.array(moe_list)
moe_arr = moe_arr.reshape(12,180,80,100)
if weighting == 'aplha_p95_scaling':
    moe_arr *= (aphro_p95)    
moe_crps = compute_crps(aphro_arr.reshape(12,180,80,30), moe_arr)

ew_arr = np.array(ew_list)
ew_arr = ew_arr.reshape(12,180,80,100)
ew_crps = compute_crps(aphro_arr.reshape(12,180,80,30), ew_arr)

moe_df['moe'] = moe_crps.flatten()
moe_df['ew'] = ew_crps.flatten()

# Save statistics
moe_df.to_csv(directory + 'bcm4rcm/data/moe_outputs/rcm_crps_' + weighting + '_' + target_experiment + '_' + 
                target_year + '_ref_' + ref_year + '.csv')