import pandas as pd
import gpflow
import sys
sys.path.append(
    '/Users/kenzatazi/Documents/CDT/Code/guepard_repo/')
sys.path.append(
    '/Users/kenzatazi/Documents/CDT/Code')
import guepard
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.special import inv_boxcox
from scipy.stats import boxcox

### Load data
dir = '/Users/kenzatazi/Documents/CDT/Code/bcm4rcm/'
df1 = pd.read_csv(
    dir + 'data/processed/hist_MPI_IITM-RegCM4-4.csv', index_col=0)
df2 = pd.read_csv(
    dir + 'data/processed/hist_MPI_REMO2009.csv', index_col=0)
df3 = pd.read_csv('data/processed/hist_CSIRO_IITM-RegCM4-4.csv', index_col=0)

df_list = []

for df in [df1, df2, df3]:
    if 'nb2' in list(df.columns):
        df = df[df['nb2']==0]
    if 'bnds' in list(df.columns):
        df = df[df['bnds'] == 0]
    df['time'] = pd.to_numeric(pd.to_datetime(df['time']))
    df['lat_group'] = pd.cut(df['lat'], 20, labels=np.arange(20))
    df['lon_group'] = pd.cut(df['lon'], 45, labels=np.arange(45))       
    gb = df.groupby(['lat_group', 'lon_group'])
    df_list.extend([gb.get_group(x)[['time', 'lon', 'lat', 'tp']].values for x in gb.groups])

arr = np.array(df_list)
# print(arr.shape)

### Process

in_list= []
in_arr_raw = arr[:, :, 0:3]
input_scaler = StandardScaler()
input_scaler.fit(in_arr_raw.reshape(-1, 3))
for i in range(len(arr)):
    in_arrs = input_scaler.transform(in_arr_raw[i])
    in_list.append(in_arrs)
in_arr = np.array(in_list)

out_arr_tr, lmbd = boxcox(arr[:, :, 3].reshape(-1)+0.001)
output_scaler = StandardScaler()
output_scaler.fit(out_arr_tr.reshape(-1, 1))
out_arr_flat = output_scaler.transform(out_arr_tr.reshape(-1, 1))
out_arr = out_arr_flat.reshape(arr.shape[0], arr.shape[1], 1,)
print(out_arr.shape)

### Model

kernel = gpflow.kernels.Matern32()
noise_var = 0.01

# list of num_split GPR models
submodels = guepard.utilities.get_gpr_submodels(
    zip(in_arr[:40], out_arr[:40]), kernel, noise_variance=noise_var)

m_rbcm = guepard.baselines.Ensemble(
    models=submodels, method=guepard.baselines.EnsembleMethods.RBCM, weighting=guepard.baselines.WeightingMethods.VAR)

opt = gpflow.optimizers.Scipy()
opt_logs = opt.minimize(m_rbcm.training_loss,
                        m_rbcm.trainable_variables, options=dict(maxiter=100))
gpflow.utilities.print_summary(m_rbcm)

### Predict

#remove duplicate rows
x_plot = np.unique(in_arr[:40].reshape(-1, 3), axis=0)
input_loc_arr = input_scaler.inverse_transform(np.array(x_plot).reshape(-1, 3))
#print(x_plot.shape)

ypred, var = m_rbcm.predict_y(x_plot)
arr= np.stack((ypred.numpy().flatten(), var.numpy().flatten()), axis=1)

# Create dataframe, apply inverse transform and calculate 95th percentile CI
df_temp= pd.DataFrame(arr, columns=['pred0', 'var0'])

df_temp['y_pred'] = output_scaler.inverse_transform(inv_boxcox(
    df_temp['pred0'].values.reshape(-1, 1), lmbd))
df_temp['95th_u'] = output_scaler.inverse_transform(inv_boxcox(
    df_temp['pred0'].values.reshape(-1, 1) + 1.96 * np.sqrt(df_temp['var0'].values.reshape(-1, 1)), lmbd))
df_temp['95th_l'] = output_scaler.inverse_transform(inv_boxcox(
    df_temp['pred0'].values.reshape(-1, 1) - 1.96 * np.sqrt(df_temp['var0'].values.reshape(-1, 1)), lmbd))
df_temp['CI'] = df_temp['95th_u'].fillna(0) - df_temp['95th_l'].fillna(0)

df_temp[['time', 'lon', 'lat']] = input_loc_arr
df_temp.set_index(['time', 'lon', 'lat'], inplace=True)

df_temp[['y_pred', '95th_u', '95th_l', 'CI']] = df_temp[[
    'y_pred', '95th_u', '95th_l', 'CI']].fillna(0)

df_temp.to_csv('data/outputs/rbcm_tr_3rcm_30year_hist.csv')