import numpy as np
import pandas as pd
import xarray as xr
import cartopy.feature as cf
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from matplotlib.patches import Patch


def scenario_plot(historical_ds, rcp45_ds, rcp85_ds, cmap='Blues'):
    """
    Plots the BCM outputs on a map.
    Args:
        df (_type_): BCM output dataframe
        scenario (str, optional): climate scenario. Defaults to "RCP8.5".
    """

    hist_ds_avg_ypred = seasonal_means(historical_ds.y_pred, 'y_pred')
    p5_annual = np.percentile(hist_ds_avg_ypred.sel(t='Annual').values, 5)
    p95_annual = np.percentile(hist_ds_avg_ypred.sel(t='Annual').values, 95)
    p5_monsoon = np.percentile(hist_ds_avg_ypred.sel(
        t='Monsoon (JJAS)').values, 5)
    p95_monsoon = np.percentile(
        hist_ds_avg_ypred.sel(t='Monsoon (JJAS)').values, 95)
    p5_winter = np.percentile(hist_ds_avg_ypred.sel(
        t='Winter (DJFM)').values, 5)
    p95_winter = np.percentile(
        hist_ds_avg_ypred.sel(t='Winter (DJFM)').values, 95)

    rcp45_ds_avg_ypred = seasonal_means(rcp45_ds.y_pred, 'y_pred')
    rcp85_ds_avg_ypred = seasonal_means(rcp85_ds.y_pred, 'y_pred')
    ds_avg_ypred = xr.concat(
        [rcp45_ds_avg_ypred, rcp85_ds_avg_ypred], pd.Index(["RCP4.5", "RCP8.5"], name='scenario'))

    proj = ccrs.PlateCarree()

    # Map
    ocean_50m = cf.NaturalEarthFeature(
        "physical", "ocean", "50m", edgecolor="darkgrey", facecolor='white')

    scenario_fg = ds_avg_ypred.plot(x="lon", y="lat", col="t", row="scenario", aspect=2, cbar_kwargs={"pad": 0.03, 'shrink': 0.8, 'label': 'BCM posterior mean [mm/day]'},
                                    subplot_kws={"projection": proj}, cmap=cmap)

    hdl = [Patch(facecolor='none', edgecolor='k', hatch='..',
                 label='< historical 5th percent.'),
           Patch(facecolor='none', edgecolor='k', hatch='//',
                 label='> historical 95th percent.')]

    for i in range(6):
        ax = scenario_fg.axs.flat[i]
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

        if i == 0:
            ax.set_title("Annual")
            gl.bottom_labels = False
            ds_avg_ypred.isel(scenario=0, t=0).plot.contourf(x='lon', y='lat', hatches=['..', '', '//'], levels=[p5_annual, p95_annual],
                                                             colors='none', add_colorbar=False,  ax=ax, transform=ccrs.PlateCarree())
        if i == 1:
            ax.set_title("Monsoon (JJAS)")
            gl.left_labels = False
            gl.bottom_labels = False
            ds_avg_ypred.isel(scenario=0, t=1).plot.contourf(x='lon', y='lat', hatches=['..', '', '//'], levels=[p5_monsoon, p95_monsoon],
                                                             colors='none', add_colorbar=False,  ax=ax, transform=ccrs.PlateCarree())
        if i == 2:
            ax.set_title("Winter (JJAS)")
            gl.left_labels = False
            gl.bottom_labels = False
            ds_avg_ypred.isel(scenario=0, t=2).plot.contourf(x='lon', y='lat', hatches=['..', '', '//'], levels=[p5_winter, p95_winter],
                                                             colors='none', add_colorbar=False,  ax=ax, transform=ccrs.PlateCarree())
        if i == 3:
            ax.set_title(" ")
            ds_avg_ypred.isel(scenario=1, t=0).plot.contourf(x='lon', y='lat', hatches=['..', '', '//'], levels=[p5_annual, p95_annual],
                                                             colors='none', add_colorbar=False,  ax=ax, transform=ccrs.PlateCarree())
        if i == 4:
            ax.set_title(" ")
            gl.left_labels = False
            ds_avg_ypred.isel(scenario=1, t=1).plot.contourf(x='lon', y='lat', hatches=['..', '', '//'], levels=[p5_monsoon, p95_monsoon],
                                                             colors='none', add_colorbar=False,  ax=ax, transform=ccrs.PlateCarree())
        if i == 5:
            ax.set_title(" ")
            gl.left_labels = False
            ds_avg_ypred.isel(scenario=1, t=2).plot.contourf(x='lon', y='lat', hatches=['..', '', '//'], levels=[p5_winter, p95_winter],
                                                             colors='none', add_colorbar=False,  ax=ax, transform=ccrs.PlateCarree())
            ax.legend(handles=hdl)

        ax.add_feature(ocean_50m)

    # Save and plot figure

    plt.savefig('plots/test_BCM_scenarios.png', bbox_inches='tight')


def mean_ci_plots(ds):

    proj = ccrs.PlateCarree()
    ds_avg_ypred = seasonal_means(ds.y_pred, 'y_pred')
    ds_avg_CI = seasonal_means(ds.CI, 'CI')

    ocean_50m = cf.NaturalEarthFeature(
        "physical", "ocean", "50m", edgecolor="darkgrey", facecolor='white')

    # Mean facet grid plot
    mean_fg = ds_avg_ypred.plot(x="lon", y="lat", col="t", sharey=True, subplot_kws={'projection': proj}, cbar_kwargs={
        "label": "BCM posterior mean [mm/day]", 'pad': 0.03})

    ax0 = mean_fg.axs.flat[0]
    gl0 = ax0.gridlines(draw_labels=True)
    gl0.top_labels = False
    gl0.right_labels = False
    ax0.add_feature(ocean_50m)

    for ax in mean_fg.axs.flat[1:]:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        ax.add_feature(ocean_50m)

    # plt.subplots_adjust(wspace=0.15)

    plt.savefig('plots/test_BCM_historical_mean.png', bbox_inches='tight')

    # CI facet grid plot
    CI_fg = ds_avg_CI.plot(x="lon", y="lat", col="t", cmap='magma', cbar_kwargs={
        "label": "BCM 95% CI [mm/day]", 'pad': 0.03}, subplot_kws={"projection": proj})

    ax0 = CI_fg.axs.flat[0]
    gl0 = ax0.gridlines(draw_labels=True)
    gl0.top_labels = False
    gl0.right_labels = False
    ax0.add_feature(ocean_50m)

    for ax in CI_fg.axs.flat[1:]:
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = False
        ax.add_feature(ocean_50m)

    plt.savefig('plots/test_BCM_historical_CI.png', bbox_inches='tight')


def seasonal_means(da: xr.DataArray, var: str) -> xr.Dataset:
    """Generate seasonal means (annual, monsoon, winter) for a given variable.

    Args:
        da (xr.DataArray): input data
        var (str): variable name

    Returns:
        xr.Dataset: seasonal means
    """

    ds_annual_avg = da.mean(dim='time')

    ds_jun = da[5::12]
    ds_jul = da[6::12]
    ds_aug = da[7::12]
    ds_sep = da[8::12]
    ds_monsoon = xr.merge([ds_jun, ds_jul, ds_aug, ds_sep])
    ds_monsoon_avg = ds_monsoon[var].mean(dim='time')

    ds_dec = da[11::12]
    ds_jan = da[0::12]
    ds_feb = da[1::12]
    ds_mar = da[2::12]
    ds_west = xr.merge([ds_dec, ds_jan, ds_feb, ds_mar])
    ds_west_avg = ds_west[var].mean(dim='time')

    ds_avg = xr.concat([ds_annual_avg, ds_monsoon_avg, ds_west_avg],
                       pd.Index(["Annual", "Monsoon (JJAS)", "Winter (DJFM)"], name='t'))

    return ds_avg


if __name__ in "__main__":
    historical_ds = xr.open_dataset('data/rbcm_test_historical_output.nc')
    rcp85_ds = xr.open_dataset('data/rbcm_test_rcp85_output.nc')
    rcp45_ds = xr.open_dataset('data/rbcm_test_rcp45_output.nc')
    mean_ci_plots(historical_ds)
    scenario_plot(historical_ds, rcp45_ds, rcp85_ds)
