import numpy as np
import pandas as pd
import xarray as xr
import cartopy.feature as cf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from shapely.geometry import LinearRing

import sys
sys.path.append('/Users/kenzatazi/Documents/CDT/Code/bcm4rcm/')

from matplotlib.patches import Patch
import matplotlib.ticker as mticker


def regional_rectangle(lonmin, lonmax, latmin, latmax, nvert=100):
    """ Return Polygon object to create regional rectangle on maps."""
    lons = np.r_[
        np.linspace(lonmin, lonmin, nvert),
        np.linspace(lonmin, lonmax, nvert),
        np.linspace(lonmax, lonmax, nvert),
    ].tolist()

    lats = np.r_[
        np.linspace(latmin, latmax, nvert),
        np.linspace(latmax, latmax, nvert),
        np.linspace(latmax, latmin, nvert),
    ].tolist()

    pgon = LinearRing(list(zip(lons, lats)))

    return pgon


def seasonal_means(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Generate seasonal means (annual, monsoon, winter) for a given variable.

    Args:
        da (xr.DataArray): input data
        var (str): variable name

    Returns:
        xr.Dataset: seasonal means
    """

    ds_annual_avg = ds.mean(dim='month')

    ds_jun = ds[5::12]
    ds_jul = ds[6::12]
    ds_aug = ds[7::12]
    ds_sep = ds[8::12]
    ds_monsoon = xr.merge([ds_jun, ds_jul, ds_aug, ds_sep])
    ds_monsoon_avg = ds_monsoon[var].mean(dim='month')

    ds_dec = ds[11::12]
    ds_jan = ds[0::12]
    ds_feb = ds[1::12]
    ds_mar = ds[2::12]
    ds_west = xr.merge([ds_dec, ds_jan, ds_feb, ds_mar])
    ds_west_avg = ds_west[var].mean(dim='month')

    ds_avg = xr.concat([ds_annual_avg, ds_monsoon_avg, ds_west_avg],
                       pd.Index([ "Annual", "Monsoon (JJAS)", "Winter (DJFM)"], name='t')) # 

    return ds_avg