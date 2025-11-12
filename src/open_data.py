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
from utils import logger

# ERA5_CHUNKS = {'time': -1, 'lat': 721 // 6, 'lon': 1440 // 12}
ERA5_SMALL_CHUNKS = {'time': -1, 'lat': 721 // 6, 'lon': 1440 // 24}
ERA5_CHUNKS_160 = {
    'time': -1,
    'lat': 721//7,
    'lon':1440//24
}

# ERA5_SMALL_CHUNKS = {'time':-1, 'lat': 361//19, 'lon':720//12} # lon o.g. 24

CHUNKS = {
    "access": {
        'small': {'time':-1, 'lat':145//5, 'lon':192//32},
        'regular': {'time':-1, 'lat':145//5, 'lon':192//8}
    }
}

best_chunks_raw = {'time':-1, 'latitude': 90, 'longitude': 120}
best_chunks = {'time':-1, 'lat': 90, 'lon': 120}


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


def open_best(chunks=None):
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
    best_ds_raw = xr.open_dataset(fname, chunks=best_chunks_raw, use_cftime=True)
    
    # Override time dimension with cftime range
    print('  -- overriding time to use cftime')
    best_ds_raw['time'] = xr.cftime_range(start='1850-01-01', freq='ME', periods=len(best_ds_raw.time.values))
    
    # Select temperature variable
    best_ds = best_ds_raw['temperature']
    # Resample to yearly mean
    print('  -- resampling to yearly mean')
    best_ds = best_ds.resample(time='YE').mean()
    # Compute and return the processed dataset
    best_ds = best_ds.compute().chunk(best_chunks if chunks is None else chunks)
    best_ds.attrs['dataset_name'] = 'best'

    return best_ds



def open_era5(
    var: str,
    return_raw: bool = False,
    save: bool = False,
    resample_method: str = "mean",
    logginglevel='ERROR',
    chunks=None
) -> xr.DataArray:
    """
    Opens the ERA5 dataset for the specified variable. It checks if the variable 
    has already been converted to Zarr format. If not, it attempts to load the 
    raw data from NetCDF files, performs resampling to yearly averages or other 
    specified methods, and returns the processed data.

    '/g/data/rt52/era5/single-levels/monthly-averaged'

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
    if not chunks: chunks = ERA5_CHUNKS
    
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



def open_era5_reanalysis(var:str , method:str) -> xr.Dataset:
    '''
    Args:
        var: the variable to be downloaded
        method: Obvilsy don't want hourly data. The method to convert to annual
    
    Some variables may need to be opened in a custom way

    I am not expecting this to be universal for all variables. May need to be manually updates for each
    variables at locations marked with # !!!!!!!!!!!!!!!
    '''
    
    from glob import glob
    path = f'/g/data/rt52/era5/single-levels/reanalysis/{var}/*/*.nc'
    files_to_open = glob(path, recursive=True)
    print(len(files_to_open))

    # Remove files later than this start year (errors before this and unaligned with other
    # ERA5 data)
    files_to_open = [f for f in files_to_open if int(f.split('/')[-2]) > 1958]
    len(files_to_open)
    
    data_ds = xr.open_mfdataset(
        files_to_open, #path,
        combine="by_coords",
        # parallel=True,
        # lock=False,  # Disable file locking (good for reach only)
        chunks=open_data.ERA5_CHUNKS, 
        use_cftime=True,
    )[var]

    # !!!!!!!!!!!!!!!
    data_ds_resample = getattr(data_ds.resample(time='YE'), method)
    
    data_ds_resample = data_ds_resample.rename({'latitude': 'lat', 'longitude': 'lon'})
    data_ds_resample = data_ds_resample.compute()
    # data_ds_resample = data_ds_resample.chunk(open_data.ERA5_CHUNKS)
    data_ds_resample = data_ds_resample.chunk({'time':-1, 'lat': 361//19, 'lon':720//24})


    MY_ERA5_PATH = os.path.join(paths.DATA_DIR, 'era5')
    SAVE_NAME = os.path.join(MY_ERA5_PATH, f'{var}.zarr')
    data_ds_resample.to_zarr(SAVE_NAME, mode='w')

    return data_ds_resample



def open_access(variable='pr', ensemble='r10i1p1f1', scenario='ssp585', resample:str='QS-DEC'):
    OPEN_KWARGS = dict(use_cftime=True, drop_variables=['lat_bnds', 'time_bnds', 'lon_bnds'],
                  chunks={'time':-1, 'lat':50})
    
    hist_ds = xr.open_dataset(
        f'/g/data/fs38/publications/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/historical/r10i1p1f1/Amon/{variable}/gn/latest/'
        f'{variable}_Amon_ACCESS-ESM1-5_historical_{ensemble}_gn_185001-201412.nc', **OPEN_KWARGS)

    ssp_p1_ds = xr.open_dataset(
        f'/g/data/fs38/publications/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/{scenario}/{ensemble}/Amon/{variable}/gn/latest/'
        f'{variable}_Amon_ACCESS-ESM1-5_{scenario}_{ensemble}_gn_201501-210012.nc', **OPEN_KWARGS)
    
    ssp_p2_ds = xr.open_dataset(
        f'/g/data/fs38/publications/CMIP6/ScenarioMIP/CSIRO/ACCESS-ESM1-5/{scenario}/{ensemble}/Amon/{variable}/gn/latest/'
        f'{variable}_Amon_ACCESS-ESM1-5_{scenario}_{ensemble}_gn_210101-230012.nc', **OPEN_KWARGS)
    
    data_raw_ds = xr.concat([hist_ds[variable], ssp_p1_ds[variable], ssp_p2_ds[variable]], dim='time')


    # Define resampling operations for each variable
    aggregation_operations = {
        'tas': 'mean',
        'tasmax': 'max',
        'tasmin': 'min',
    }
    
    if variable == 'pr':
        # Special handling for precipitation
        data_ds = data_raw_ds * 86400  # Convert m/s to mm/day
        data_ds = data_ds.resample(time=resample).sum().compute()
        # Remove negative values (set to 0)
        data_ds = data_ds.where(data_ds >= 0, 0)
    elif variable in aggregation_operations:
        # Use dynamic method selection for other variables
        agg_func = aggregation_operations[variable]
        print(f'{agg_func=}')
        data_ds = getattr(data_raw_ds.resample(time=resample), agg_func)()
    else:
        raise ValueError(f"Unsupported variable: {variable}")

    # The last year isn't correct. The sum QS-DEC bug this. Just removing both though
    data_ds = data_ds.sel(time=data_ds.time.dt.year< 2299)

    data_ds.attrs['dataset_name'] = f'access_{scenario}_{ensemble}'
    data_ds.attrs['resample'] = resample

    return data_ds


def chunk_lat_lon(ds, num_chunks, lat_name='lat', lon_name='lon'):
    """
    Calculate chunk sizes for a dataset to divide lat and lon into approximately `num_chunks` total chunks,
    while keeping the 'time' dimension unchunked.

    Args:
        ds (xr.Dataset): The input dataset.
        num_chunks (int): The desired total number of chunks for lat and lon combined.
        lat_name (str): The name of the latitude dimension in the dataset. Default is 'lat'.
        lon_name (str): The name of the longitude dimension in the dataset. Default is 'lon'.

    Returns:
        dict: A dictionary of chunk sizes for 'time', 'lat', and 'lon'.
    """
    # Calculate the square root of the total number of chunks
    num_chunks_sqrt = int(np.sqrt(num_chunks))

    # Get the sizes of the latitude and longitude dimensions
    dim_sizes = ds.sizes

    # Calculate chunk sizes for latitude and longitude
    lat_chunk_size = dim_sizes[lat_name] // num_chunks_sqrt
    lon_chunk_size = dim_sizes[lon_name] // num_chunks_sqrt

    # Define chunk sizes for the dataset
    chunks = {'time': -1, lat_name: lat_chunk_size, lon_name: lon_chunk_size}
    
    return chunks