"""
Open and process datasets for analysis.

This file contains functions to open and process GPCC precipitation and BEST temperature datasets.
"""


import os, sys
import numpy as np
import xarray as xr

sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))
import paths

import utils
logger = utils.get_notebook_logger()

ERA5_CHUNKS = {'time': -1, 'lat': 721 // 6, 'lon': 1440 // 12}


def rechunk_lat_lon(dataset: xr.Dataset, lat_chunk_size: int, lon_chunk_size: int) -> xr.Dataset:
    """
    Rechunk the dataset to divide the 'lat' and 'lon' dimensions into specified sizes
    while keeping 'time' as a single chunk.

    Args:
        dataset (xr.Dataset): The input dataset to rechunk.
        lat_chunk_size (int): Desired chunk size for the 'lat' dimension.
        lon_chunk_size (int): Desired chunk size for the 'lon' dimension.

    Returns:
        xr.Dataset: The rechunked dataset.
    """
    # Calculate the total sizes of 'lat' and 'lon'
    lat_size = dataset.sizes['lat']
    lon_size = dataset.sizes['lon']

    # Ensure the chunk sizes don't exceed the total size of the dimension
    lat_chunk_size = min(lat_chunk_size, lat_size)
    lon_chunk_size = min(lon_chunk_size, lon_size)

    # Rechunk the dataset
    rechunked_dataset = dataset.chunk({
        'time': -1,  # Keep 'time' as a single chunk
        'lat': lat_chunk_size,
        'lon': lon_chunk_size
    })
    
    return rechunked_dataset

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
        print(f'  -- resampling to {resample} sum')
        gcpp_ds = gcpp_raw_ds.resample(time=resample).sum().compute()
        # Mask out non-finite values
        print('  -- masking out non-finite values')
        gcpp_ds = gcpp_ds.where(np.isfinite(gcpp_raw_ds.isel(time=0).drop('time')))
    else:
        gcpp_ds = gcpp_raw_ds
    # Lats are wrong way around for some reason
    gcpp_ds = gcpp_ds.sortby('lat')
    
    gcpp_ds.attrs['dataset_name'] = 'gpcc'
    
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
    best_ds.attrs['dataset_name'] = 'best'

    return best_ds



def open_era5(
    var: str,
    return_raw: bool = False,
    save: bool = False,
    resample_method: str = "mean",
    logginglevel='ERROR'
) -> xr.DataArray:
    """
    Opens the ERA5 dataset for the specified variable. It checks if the variable 
    has already been converted to Zarr format. If not, it attempts to load the 
    raw data from NetCDF files, performs resampling to yearly averages or other 
    specified methods, and returns the processed data.

    Args:
        var (str): The name of the ERA5 variable to load (e.g., 't2m', 'cape').
        return_raw (bool): Return the raw data (don't resample and rename).
        save (bool): If I want the dataset saved to Zarr or not in the local ERA5 directory.
        resample_method (str): Resampling method to apply ('mean', 'max', 'sum'). Defaults to 'mean'.

    Returns:
        xr.DataArray: The processed ERA5 data for the specified variable, resampled as specified.

    Example:
        data_ds = open_era5('cape', save=True, resample_method='max')
    """
    utils.change_logginglevel(logginglevel)
    chunks = ERA5_CHUNKS
    
    MY_ERA5_PATH = os.path.join(paths.DATA_DIR, 'era5')
    
    # List the variables that have already been converted to Zarr format
    ERA5_SAVE_VARIABLES = list(map(lambda x: x.split('.')[0], os.listdir(MY_ERA5_PATH)))

    # If the variable is already available as a Zarr file, load and return it
    if var in ERA5_SAVE_VARIABLES and not return_raw:
        logger.info(' - Variable already converted to zarr')
        data_ds = xr.open_zarr(os.path.join(MY_ERA5_PATH, f'{var}.zarr'))
        save = False  # If variable already saved, override save
        
    else: 
        logger.info(f'New Variable - attempting to open {var} from {paths.ERA5_PATH}')

        full_path = os.path.join(paths.ERA5_PATH, var, '*', '*.nc')
        
        logger.debug(f'{full_path=}')
        
        data_raw_ds = xr.open_mfdataset(full_path, chunks=chunks)
        if return_raw:
            logger.info('Returning raw data')
            return data_raw_ds

        logger.debug(f' - Resampling to yearly {resample_method}')
        if resample_method == "mean":
            data_ds = data_raw_ds.resample(time='YE').mean()
        elif resample_method == "max":
            data_ds = data_raw_ds.resample(time='YE').max()
        elif resample_method == "sum":
            data_ds = data_raw_ds.resample(time='YE').sum()
        else:
            raise ValueError(f"Unsupported resampling method: {resample_method}")

    if 'latitude' in list(data_ds.coords):
        logger.info('Renaming latitude - lat and longitude - lon')
        data_ds = data_ds.rename({'latitude': 'lat', 'longitude': 'lon'})

    data_ds = data_ds[var]
    data_ds.attrs['dataset_name'] = 'era5'
    data_ds.attrs['resample_method'] = resample_method


    if save:
        SAVE_NAME = os.path.join(MY_ERA5_PATH, f'{var}.zarr')
        logger.info(f' - Saving - {SAVE_NAME=}')
        data_ds.to_zarr(SAVE_NAME, mode='w')

    data_ds = data_ds.chunk(chunks)

    return data_ds




def open_access_precip(resample:str='QS-DEC'):
    OPEN_KWARGS = dict(use_cftime=True, drop_variables=['lat_bnds', 'time_bnds', 'lon_bnds'],
                  chunks={'time':-1, 'lat':50})
    
    hist_ds = xr.open_dataset(
        '/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r10i1p1f1/Amon/pr/gn/latest/'
        'pr_Amon_ACCESS-ESM1-5_historical_r10i1p1f1_gn_185001-201412.nc', **OPEN_KWARGS)

    ssp585_p1_ds = xr.open_dataset(
        '/g/data/fs38/publications/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp585/r10i1p1f1/Amon/pr/gn/latest/'
        'pr_Amon_ACCESS-ESM1-5_ssp585_r10i1p1f1_gn_201501-210012.nc', **OPEN_KWARGS)
    
    ssp585_p2_ds = xr.open_dataset(
        '/g/data/fs38/publications/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/ssp585/r10i1p1f1/''Amon/pr/gn/latest/'
        'pr_Amon_ACCESS-ESM1-5_ssp585_r10i1p1f1_gn_210101-230012.nc', **OPEN_KWARGS)
    
    data_raw_ds = xr.concat([hist_ds['pr'], ssp585_p1_ds['pr'], ssp585_p2_ds['pr']], dim='time')

    # m/s -> mm/day
    data_ds = data_raw_ds*86400
    
    data_ds = data_ds.resample(time=resample).sum().compute()
    # data_ds = data_ds.where(data_ds.time.dt.month == 12, drop=True)
    
    # There are negative values for some reason. For now just remove them and move on.
    data_ds = data_ds.where(data_ds >=0, data_ds, 0)
    
    # The last year isn't correct. The sum QS-DEC bug this. Just removing both though
    data_ds = data_ds.sel(time=data_ds.time.dt.year< 2299)

    # gcpp_ds = gcpp_raw_ds.resample(time=resample).sum().compute()

    
    data_ds.attrs['dataset_name'] = 'access'


    return data_ds
