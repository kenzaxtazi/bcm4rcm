import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot(bcm_df: pd.DataFrame):

    bcm_df['time'] = pd.to_datetime(bcm_df['time'])
    bcm_df_monthly = bcm_df.groupby(
        [bcm_df['lon'], bcm_df['lat'], bcm_df['time'].dt.month]).mean()
    bcm_df_monthly.drop(columns=['time'], inplace=True)
    bcm_df_monthly.reset_index(inplace=True)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

    for i in range(4):
        s = i*12
        ax[i].plot(bcm_df_monthly.time[s:s+12], bcm_df_monthly.y_pred[s:s+12])
        ax[i].fill_between(bcm_df_monthly.time[s:s+12],
                           bcm_df_monthly['95th_l'][s:s+12],  bcm_df_monthly['95th_u'][s:s+12], alpha=0.2)
        ax[i].set_title(str(bcm_df_monthly.lat[s])+'°N ' +
                        str(bcm_df_monthly.lon[s]) + '°E')
        ax[i].set_xticks(np.arange(2, 13, 2), ['F', 'A', 'J', 'A', 'O', 'D'])
        ax[i].set_xlabel('Month')

    ax[0].set_ylabel('Precipiation [mm/day]')
    plt.savefig('plots/loc_dist_test_2rcm_1961.png',
                dpi=300, bbox_inches='tight')


if __name__ in '__main__':

    bcm_df = pd.read_csv('data/outputs/rbcm_tr_2rcm_1961_locs.csv')
    plot(bcm_df)
