
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def high_precip(df: pd.DataFrame):
    """
    Returns PDF and CDF plots of the BCM output focused on high precipitation tail of the distribution.

    Args:
        df (pd.DataFrame): BCM output dataframe
    """

    threshold = 15

    # sharey to have the same y axis limits, wspace=0 to make the axes connect
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # x limits of the axes
    xlim1 = (5, 35)

    # sharey to have the same y axis limits, wspace=0 to make the axes connect
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # x limits of the axes
    xlim1 = (5, 35)

    # calculate the histogram bin width such that it is equal for both axes
    # we choose 50 bins per axis, you can change this value
    bins1 = np.arange(xlim1[0], xlim1[1] + 1, np.diff(xlim1) / 40)

    # plot the same data in both axes
    ax1.hist(df['historical'], bins=bins1, edgecolor='b', density=True,
             histtype='step', label='historical', fill=False)
    ax1.hist(df['RCP4.5'], bins=bins1, label='RCP4.5', density=True, edgecolor='g',
             histtype='step', fill=False)
    ax1.hist(df['RCP8.5'], bins=bins1, label='RCP8.5', edgecolor='r', density=True,
             histtype='step', fill=False)

    ax2.hist(df['historical'], bins=bins1, edgecolor='b', density=True,
             histtype='step', label='historical', cumulative=True, fill=False)
    ax2.hist(df['RCP4.5'], bins=bins1, label='RCP4.5', density=True, edgecolor='g',
             histtype='step', cumulative=True, fill=False)
    ax2.hist(df['RCP8.5'], bins=bins1, label='RCP8.5', edgecolor='r', density=True,
             histtype='step', cumulative=True, fill=False)

    # plot the same data in both axes

    # set the x limits
    ax1.set_xlim(xlim1)
    ax2.set_xlim(xlim1)

    # add line to indicate the change in x scale
    ax1.axvline(threshold, clip_on=False, color='lightblue',
                linestyle='--', label='high precipitation threshold')
    ax2.axvline(threshold, clip_on=False, color='lightblue',
                linestyle='--', label='high precipitation threshold')

    leg = ax2.legend()
    leg.get_frame().set_facecolor('white')

    ax1.set_ylabel('Cumulative distribution function')
    ax2.set_ylabel('Probability distribution function')
    ax1.set_xlabel('Precipitation (mm/day)')
    plt.savefig('plots/high_precip_dist.png', bbox_inches='tight')


def low_precip(df: pd.DataFrame):
    """
    Returns PDF and CDF plots of the BCM output focused on low precipitation tail of the distribution.

    Args:
        df (pd.DataFrame): BCM output dataframe
    """

    threshold = 0.5

    # sharey to have the same y axis limits, wspace=0 to make the axes connect
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # x limits of the axes
    xlim1 = (0, 2)

    # calculate the histogram bin width such that it is equal for both axes
    # we choose 50 bins per axis, you can change this value
    bins1 = np.arange(xlim1[0], xlim1[1] + 1, np.diff(xlim1) / 40)

    # plot the same data in both axes
    ax1.hist(df['historical'], bins=bins1, edgecolor='b', density=True,
             histtype='step', label='historical', fill=False)
    ax1.hist(df['RCP4.5'], bins=bins1, label='RCP4.5', density=True, edgecolor='g',
             histtype='step', fill=False)
    ax1.hist(df['RCP8.5'], bins=bins1, label='RCP8.5', edgecolor='r', density=True,
             histtype='step', fill=False)

    ax2.hist(df['historical'], bins=bins1, edgecolor='b', density=True,
             histtype='step', label='historical', cumulative=True, fill=False)
    ax2.hist(df['RCP4.5'], bins=bins1, label='RCP4.5', density=True, edgecolor='g',
             histtype='step', cumulative=True, fill=False)
    ax2.hist(df['RCP8.5'], bins=bins1, label='RCP8.5', edgecolor='r', density=True,
             histtype='step', cumulative=True, fill=False)

    # set the x limits
    ax1.set_xlim(xlim1)
    ax2.set_xlim(xlim1)

    # add line to indicate the change in x scale
    ax1.axvline(threshold, clip_on=False, linestyle=':',
                label='low precipitation threshold')
    ax2.axvline(threshold, clip_on=False, linestyle=':',
                label='low precipitation threshold')

    leg = ax2.legend()
    leg.get_frame().set_facecolor('white')

    ax1.set_ylabel('Cumulative distribution function')
    ax2.set_ylabel('Probability distribution function')

    ax1.set_xlabel('Precipitation [mm/day]')
    ax2.set_xlabel('Precipitation [mm/day]')

    plt.savefig('plots/low_precip_dist.png', bbox_inches='tight')


for __name__ in '__main__':
    df = pd.read_csv('data/rbcm_single_rcm_samples.csv')
    low_precip(df)
    high_precip(df)
