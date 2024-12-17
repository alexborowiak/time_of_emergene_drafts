"""
Open and process datasets for analysis.

This file contains functions to open and process GPCC precipitation and BEST temperature datasets.
"""


import os, sys

import numpy as np
import xarray as xr

sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))

import paths

def open_gpcc(resample=False):
    """
    Open and process GPCC precipitation dataset.

    Returns:
        xarray.Dataset: Processed GPCC dataset with yearly sum.
    """
    PRECIP_PATH = '/g/data/w40/ab2313/time_of_emergence/GPCC/precip.mon.total.v7.nc'
    print(f'Opening GPCC dataset from - {PRECIP_PATH}')
    # Open dataset with chunking for efficient processing
    gcpp_raw_ds = xr.open_dataset(PRECIP_PATH, chunks={'time':-1, 'lat':120, 'lon':180}).precip

    if resample:
        # Resample to yearly sum
        print('  -- resampling to yearly sum')
        gcpp_ds = gcpp_raw_ds.resample(time='Y').sum().compute()
        # Mask out non-finite values
        print('  -- masking out non-finite values')
        gcpp_ds = gcpp_ds.where(np.isfinite(gcpp_raw_ds.isel(time=0).drop('time')))
    else:
        gcpp_ds = gcpp_raw_ds
    # Lats are wrong way around for some reason
    gcpp_ds = gcpp_ds.sortby('lat')
    return gcpp_ds


def open_best():
    """
    Open and process BEST temperature dataset.

    Returns:
        xarray.Dataset: Processed BEST dataset with yearly mean temperature.
    """
    # Define file path
    ROOT_DIR = '/g/data/w40/ab2313/time_of_emergence'

    fname = os.path.join(ROOT_DIR, 'best', 'Land_and_Ocean_LatLong1_time_chunk.zarr')
    print(f'Opening best dataset from - {fname}')
    # Open dataset with chunking and cftime support
    best_ds_raw = xr.open_dataset(fname, chunks={'time':-1, 'latitude': 90, 'longitude': 120}, use_cftime=True)
    
    # Override time dimension with cftime range
    print('  -- overriding time to use cftime')
    best_ds_raw['time'] = xr.cftime_range(start='1850-01-01', freq='M', periods=len(best_ds_raw.time.values))
    
    # Select temperature variable
    best_ds = best_ds_raw['temperature']
    # Resample to yearly mean
    print('  -- resampling to yearly mean')
    best_ds = best_ds.resample(time='Y').mean()
    # Compute and return the processed dataset
    best_ds = best_ds.compute()
    return best_ds



def open_era5(var: str, return_raw:bool=False, save:bool=False) -> xr.DataArray:
    """
    Opens the ERA5 dataset for the specified variable. It checks if the variable 
    has already been converted to Zarr format. If not, it attempts to load the 
    raw data from NetCDF files, performs resampling to yearly averages, and returns 
    the processed data.

    Args:
        var (str): The name of the ERA5 variable to load (e.g., 't2m', 'cape').
        return_raw (bool): Return the raw data (don't resample and rename)
        save (bool): If I want the dataset save to zarr or not in local era5 directory

    Returns:
        xr.DataArray: The processed ERA5 data for the specified variable, resampled to yearly averages.

    Example:
    data_ds = open_data.open_era5('cape', save=True)
    # This will return 'cape'
    
    # This can now be run as
    data_ds = open_data.open_era5('cape', save=True)
    
    # If erorrs occur, this can be used to get the raw data
    data_ds = open_data.open_era5('cape', return_raw=True)
    """
    MY_ERA5_PATH = os.path.join(paths.DATA_DIR, 'era5')
    
    # List the variables that have already been converted to Zarr format
    ERA5_SAVE_VARIABLES = list(map(lambda x: x.split('.')[0], os.listdir(MY_ERA5_PATH)))

    # If the variable is already available as a Zarr file, load and return it
    # If I want raw, then don't do this
    if var in ERA5_SAVE_VARIABLES and not return_raw:
        print(' - Variable already converted to zarr')
        data_ds = xr.open_zarr(os.path.join(MY_ERA5_PATH, f'{var}.zarr'))
        save=False # If variable already save- orverride save
        
    else: 
        print(f'New Variables - attempting to open {var} from {paths.ERA5_PATH}')
        
        data_raw_ds = xr.open_mfdataset(
            os.path.join(paths.ERA5_PATH, var, '*', '*.nc'), 
            chunks={'time': -1, 'lat': 721 // 6, 'lon': 1440 // 12}
        )
        if return_raw:
            print('Returning raw data')
            return data_raw_ds
        print(' - Resample to yearly mean')
        data_ds = data_raw_ds.resample(time='YE').mean()

    if 'latitude' in list(data_ds.coords):
        print('Renaming latitutde - lat and longitude - lon')
        data_ds = data_ds.rename({'latitude': 'lat', 'longitude': 'lon'})

    if save:
        SAVE_NAME = os.path.join(MY_ERA5_PATH, f'{var}.zarr')
        print(f' - Saving - {SAVE_NAME=}')
        data_ds.to_zarr(SAVE_NAME, mode='w')

    return data_ds[var]
