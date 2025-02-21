""" Generate MoE predictions for given period and experiment"""

# Import libraries
import os
import json
import glob
import tqdm
import warnings

import scipy as sp
import numpy as np
import pandas as pd

# Set directory
DIR = json.load(open("../../config.json"))["code_dir"]
warnings.simplefilter(action="ignore", category=FutureWarning)


#############################################

target_year = "1976_2005"
target_experiment = "historical"
ref_year = "1951_2005"
weighting = "alpha"

"""
target_year = os.environ['target_year']
ref_year =  os.environ['ref_year']
target_experiment =  os.environ['target_experiment']
weighting = os.environ['weighting']
print(weighting)
"""
##############################################


# Identify data

# Get BCM filenames
bcm_path = (
    DIR
    + "bcm4rcm/data/bcm_outputs/"
    + target_experiment
    + "/bcm_"
    + target_experiment
    + "_*"
    + target_year
    + ".csv"
)
print(bcm_path)
bcm_files = sorted(glob.glob(bcm_path))

lambda_path = (
    DIR
    + "bcm4rcm/data/bcm_outputs/"
    + target_experiment
    + "/lambda_"
    + target_experiment
    + "_*"
    + target_year
    + ".npy"
)
print(lambda_path)
lambda_files = sorted(glob.glob(lambda_path))


# Get weights
if weighting == "alpha":
    weight_path = (
        DIR
        + "bcm4rcm/data/weights/250110_weights_beta_p95_for_wass_no_p95_for_dist_5x5_1951_2005.npy"
    )
    print(weight_path)
    alpha_arr = np.load(weight_path)

# Load data and prepare data
np.random.seed(42)

ref_df = pd.read_csv(bcm_files[0], index_col=0).reset_index()
ref_df.sort_values(["month", "lon", "lat"], inplace=True)
moe_df = ref_df[["month", "lon", "lat"]]

# Number of total samples
n = int(1e5)

moe_sample_list = []
ew_sample_list = []

# Loop over RCMs
for j in tqdm.tqdm(range(len(bcm_files))):
    bcm_df = pd.read_csv(bcm_files[j], index_col=0).reset_index()
    bcm_df.sort_values(["month", "lon", "lat"], inplace=True)
    lmbda = np.load(lambda_files[j])

    # Generate sample from each BCM
    mu = bcm_df["mean"].values
    std = np.sqrt(bcm_df["var"].values)

    # Generate MoE samples
    weights = alpha_arr[j].flatten()
    weights = np.nan_to_num(weights, 0)
    moe_samples = [
        np.random.normal(m, s, int(np.ceil(n * w))) for m, s, w in zip(mu, std, weights)
    ]
    moe_samp = [
        np.nan_to_num(sp.special.inv_boxcox(ms, lmbda), 0) for ms in moe_samples
    ]
    moe_sample_list.append(moe_samp)

    # Generate EW samples
    ew_samples = [np.random.normal(m, s, int(np.ceil(n / 13))) for m, s in zip(mu, std)]
    ew_samp = [np.nan_to_num(sp.special.inv_boxcox(ms, lmbda), 0) for ms in ew_samples]
    ew_sample_list.append(ew_samp)


# MoE statistics
moe_list = [
    np.concatenate([moe_sample_list[j][i] for j in range(13)])[:n]
    for i in range(12 * 180 * 80)
]

for i in range(len(moe_list)):
    if len(moe_list[i]) == 0:
        moe_list[i] = np.ones(n) * np.nan

moe_arr = np.array(moe_list)
moe_df["moe_median"] = np.median(moe_arr, axis=-1)
moe_df["moe_p95"] = np.percentile(moe_arr, 95, axis=-1)
moe_df["moe_p5"] = np.percentile(moe_arr, 5, axis=-1)


# EW statistics
ew_list = [
    np.concatenate([ew_sample_list[j][i] for j in range(13)])[:n]
    for i in range(12 * 180 * 80)
]

for i in range(len(ew_list)):
    if len(ew_list[i]) == 0:
        ew_list[i] = np.ones(n) * np.nan

ew_arr = np.array(ew_list)
moe_df["ew_p5"] = np.percentile(ew_arr, 5, axis=-1)
moe_df["ew_p95"] = np.percentile(ew_arr, 95, axis=-1)
moe_df["ew_median"] = np.median(ew_arr, axis=-1)

# Save statistics
moe_df.to_csv(
    DIR
    + "bcm4rcm/data/moe_outputs/moe_"
    + weighting
    + "_"
    + target_experiment
    + "_"
    + target_year
    + "_ref_"
    + ref_year
    + ".csv"
)
