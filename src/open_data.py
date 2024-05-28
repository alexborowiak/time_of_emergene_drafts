"""
Open and process datasets for analysis.

This file contains functions to open and process GPCC precipitation and BEST temperature datasets.
"""


import os

import numpy as np
import xarray as xr

def open_gpcc(resample=False):
    """
    Open and process GPCC precipitation dataset.

    Returns:
        xarray.Dataset: Processed GPCC dataset with yearly sum.
    """
    PRECIP_PATH = '/g/data/w40/ab2313/PhD/time_of_emergence/GPCC/precip.mon.total.v7.nc'
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
    ROOT_DIR = '/g/data/w40/ab2313/PhD/time_of_emergence'

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