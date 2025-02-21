"""
Independently calculating the Wasserstein distance between the BCM samples and the APHRODITE data.
25th September 2024
"""

# Import libraries
import os
import sys
import json
import tqdm
import scipy as sp
import numpy as np
import pandas as pd

# Custom libraries
DIR = json.load(open("../../config.json"))["code_dir"]
sys.path.append(DIR)
from load import aphrodite  # noqa

#############################################

"""
rcm = 'CSIRO_RegCM4'
year = '1951_1980'
experiment = 'historical'
"""
rcm = os.environ["rcm"]
year = os.environ["year"]
experiment = os.environ["experiment"]

##############################################

# Get filenames
ref = experiment + "_" + rcm + "_" + year
bcm_file = DIR + "bcm4rcm/data/bcm_outputs/" + experiment + "/bcm_" + ref + ".csv"
print(bcm_file)

lambda_file = DIR + "bcm4rcm/data/bcm_outputs/" + experiment + "/lambda_" + ref + ".npy"
print(lambda_file)

# Load data
bcm_df = pd.read_csv(bcm_file, index_col=0).reset_index()
lmbda = np.load(lambda_file)

# Load processed data (the final implementation does not use the p95 values to scale the data)
# rcm_df = pd.read_csv(DIR + 'bcm4rcm/data/processed/' + experiment + '/' + ref + '.csv')
# p95 = np.percentile(m_df['tp'].values, 95)

# Load APHRODITE data
years = year.split("_")
aphro_ds = aphrodite.collect_APHRO("hma", minyear=years[0], maxyear=years[1])
aphro_df = aphro_ds.to_dataframe().reset_index()
aphro_df["month"] = aphro_df["time"].dt.month
aphro_df.sort_values(by=["month", "lon", "lat"], inplace=True)

# Make dataframe for Wasserstein distance
bcm_df.sort_values(by=["month", "lon", "lat"], inplace=True)
wass_df = bcm_df[["month", "lon", "lat"]]
wass_df["wass"] = np.zeros(len(wass_df))
# wass_df['p95'] = np.zeros(len(wass_df))

# Location loop
bcm_sample_list_tr = []

for i in tqdm.tqdm(range(len(bcm_df))):

    # Get BCM samples
    mu = bcm_df.loc[i, "mean"]
    var = bcm_df.loc[i, "var"]
    lat = bcm_df.loc[i, "lat"]
    lon = bcm_df.loc[i, "lon"]
    mon = bcm_df.loc[i, "month"]

    bcm_samples_raw = np.random.normal(loc=mu, scale=np.sqrt(var), size=100)
    bcm_samples_tr = sp.special.inv_boxcox(bcm_samples_raw, lmbda)
    bcm_samples_tr = np.nan_to_num(bcm_samples_tr, nan=0)
    # p95 = np.percentile(bcm_samples_tr, 95)  #.flatten(), 95)
    bcm_sample_list_tr.append(bcm_samples_tr)  # /p95)
    # wass_df.loc[i, 'p95'] = p95

# aphro_p95_df = bcm_df[['month', 'lon', 'lat']]
# aphro_p95_df['p95'] = np.zeros(len(aphro_p95_df))

for i in tqdm.tqdm(range(len(bcm_df))):
    bcm_samples = bcm_sample_list_tr[i]
    # Get Aphrodite
    loc_aphro_df = aphro_df[
        (aphro_df["lat"] == lat) & (aphro_df["lon"] == lon) & (aphro_df["month"] == mon)
    ]
    aphro_samples = loc_aphro_df["tp"].values
    # aphro_p95 = np.percentile(aphro_samples_raw, 95)
    # aphro_samples = aphro_samples_raw/aphro_p95
    # aphro_p95_df.loc[i, 'p95'] = aphro_p95

    # Calulate Wasserstein Distance
    wass_dist = sp.stats.wasserstein_distance(bcm_samples, aphro_samples)
    wass_df.loc[i, "wass"] = wass_dist

wass_df.to_csv(DIR + "bcm4rcm/data/weights/wass_" + ref + ".csv")
aphro_df.to_csv(DIR + "bcm4rcm/data/weights/aphro_p95.csv")
