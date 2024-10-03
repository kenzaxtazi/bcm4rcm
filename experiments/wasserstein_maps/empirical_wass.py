#!/usr/bin/env python
# coding: utf-8

# # Independently calculating the mean and variance for each RCM
# 25th September 2024

# Import libraries
import os
import sys
import tqdm
import scipy as sp
import numpy as np
import pandas as pd

# Custom libraries
directory = '/data/hpcdata/users/kenzi22/'
#directory = '/Users/kenzatazi/Documents/CDT/Code/'

sys.path.append(directory + 'bcm4rcm/')
sys.path.append(directory)
from load import aphrodite

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

# Get filenames 
ref = experiment +  '_' + rcm + '_' + year
bcm_file = directory + 'bcm4rcm/data/bcm_outputs/' + experiment + '/bcm_' + ref + '.csv'
print(bcm_file)

lambda_file = directory + 'bcm4rcm/data/bcm_outputs/' + experiment + '/lambda_' + ref + '.npy'
print(lambda_file)


# Load data
bcm_df = pd.read_csv(bcm_file, index_col=0).reset_index()
lmbda = np.load(lambda_file)


# Load Aphrodite data
aphro_ds = aphrodite.collect_APHRO('hma', minyear='1976', maxyear='2005')
aphro_df = aphro_ds.to_dataframe().reset_index()
aphro_df['month'] = aphro_df['time'].dt.month

# Make dataframe for Wasserstein distance
wass_df = bcm_df[['lat', 'lon', 'month']]
wass_df['wass'] = np.zeros(len(wass_df))


# Location loop

for i in tqdm.tqdm(range(len(bcm_df))):
    
    # Get BCM samples
    mu = bcm_df.loc[i, 'mean']
    var = bcm_df.loc[i, 'var']
    lat = bcm_df.loc[i, 'lat']
    lon = bcm_df.loc[i, 'lon']
    mon = bcm_df.loc[i, 'month']

    bcm_samples_raw = np.random.normal(loc=mu, scale=np.sqrt(var), size=100)
    bcm_samples_tr = sp.special.inv_boxcox(bcm_samples_raw, lmbda)
    bcm_samples_tr = np.nan_to_num(bcm_samples_tr, nan=0)
    p95 = np.percentile(bcm_samples_tr, 95)
    bcm_samples = bcm_samples_tr/p95

    # Get Aphrodite 
    loc_aphro_df = aphro_df[(aphro_df['lat'] == lat) & (aphro_df['lon']== lon) & (aphro_df['month'] == mon)]
    aphro_samples_raw = loc_aphro_df['tp'].values
    aphro_samples = aphro_samples_raw/np.percentile(aphro_samples_raw, 95)

    # Calulate Wasserstein Distance
    wass_dist = sp.stats.wasserstein_distance(bcm_samples, aphro_samples)
    wass_df.loc[i, 'wass'] = wass_dist

wass_df.to_csv(directory + 'bcm4rcm/data/weights/wass_' + ref +'.csv')