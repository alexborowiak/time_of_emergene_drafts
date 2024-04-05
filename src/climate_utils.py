"""
climate_utils.py - Utility functions for climate data analysis.

This module contains utility functions related to climate data processing and analysis.
"""
import numpy as np
import xarray as xr

def weighted_lat_lon_mean(dataset: xr.Dataset):
    """
    Calculate the weighted mean of a dataset based on latitude and longitude.
    
    Parameters:
        dataset (xarray.Dataset): The dataset for which to calculate the weighted mean.
    
    Returns:
        xarray.Dataset: The dataset with the weighted mean calculated along 'lat' and 'lon' dimensions.
    """
    weights = np.cos(np.deg2rad(dataset.lat))
    weights.name = 'weights'

    # Calculating the weighted mean.
    wmean_dataset = dataset.weighted(weights).mean(dim=['lat', 'lon'])

    return wmean_dataset