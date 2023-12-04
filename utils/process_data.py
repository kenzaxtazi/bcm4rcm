import xarray as xr
import sys #noqa
sys.path.append( '/Users/kenzatazi/Documents/CDT/Code') #noqa
from load import cordex 


# Possible experiments: 'historical', 'rcp45', 'rcp85'
# Possible time periods: '1976-2005', '2036-2065', '2066-2095'

domain = 'WAS'
model_dict = {'CanESM2':['CanESM2', 'RegCM4'], 'NOAA_IITM':['NOAA', 'RegCM4'], 'CNRM_RegCM4': ['CNRM', 'RegCM4'], 
              'MPI_IITM': ['MPI', 'RegCM4'], 'IPSL_RegCM4': ['IPSL', 'RegCM4'], 'CSIRO': ['CSIRO', 'RegCM4'],
              'EC-EARTH':['EC-EARTH', 'RCA4'], 'MIROC': ['MIROC','RCA4'], 'NOAA_SMHI':['NOAA', 'RCA4'],
              'CNRM': ['CNRM', 'RCA4'], 'MPI_RCA4': ['MPI', 'RCA4'], 'IPSL_RCA4': ['IPSL', 'RCA4'],
              'MPI-REMO2009':['MPI', 'REMO2009']}


def process_data(model_list: list, experiment: str, minyear: str, maxyear: str):
    """
    Create CSV files for each model for historical period from netcdf files

    Args:
        model_list (list): _description_
        experiment (str): _description_
    """
    for model in model_list:
        model_ds = cordex.collect_CORDEX(domain, minyear, maxyear, experiment, 
                                         gcm_model=model_dict[model][0], rcm_model=model_dict[model][1])
        sliced_ds = model_ds.sel(lat=slice(20, 40), lon=slice(60, 105))
        sliced_ds = sliced_ds.drop_vars('time_bnds') 
        #print(sliced_ds)
        if 'nb2' in sliced_ds.dims:
            clean_ds = sliced_ds.isel(nb2=0)
        elif 'bnds' in sliced_ds.dims:
            clean_ds = sliced_ds.isel(bnds=0)
        else:
            clean_ds = sliced_ds
        df = clean_ds.to_dataframe().reset_index()
        df.to_csv('data/processed/'+ experiment + '_' + model + '_' + minyear + '_' + maxyear + '.csv')