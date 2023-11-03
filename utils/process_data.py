import xarray as xr


def process_data(ds):
    ds_sel = ds.sel(lat=slice(20, 40), lon=slice(60, 105))
    if 'nb2' in da_sel.dims:
        da_sel_clean = da_sel.isel(nb2=0).drop_vars('time_bnds')
    else:
        da_sel_clean = da_sel.isel(bnds=0).drop_vars('time_bnds')
    df = da_sel_clean.to_dataframe().reset_index()
    paths = path.split('/')
    df.to_csv('data/processed/' + paths[-1][:-2]+'csv')
    return df
