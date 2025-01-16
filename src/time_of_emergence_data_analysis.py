'''
-------------------- Misc
These functions aren't for getting a metric or ToE
Bur rather are doing something specific themselves
'''

import os, sys
from typing import Tuple, List
from itertools import combinations


import numpy as np
import pandas as pd
import xarray as xr
import dask.array as daskarray

from numpy.typing import ArrayLike

# My imports
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts'))
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))

import toe_constants as toe_const
import utils
logger = utils.get_notebook_logger()


def create_exceedance_single_point_dict(toe_ds, timeseries_ds):
    """
    Creates a dictionary with year, corresponding datetime, and value from two datasets.

    Parameters:
        toe_ds (xarray.Dataset): Dataset containing a single value representing a year.
        timeseries_ds (xarray.Dataset): Dataset containing a time series.

    Returns:
        dict: A dictionary with keys 'year', 'year_datetime', and 'val'.

    Note:
        This function assumes both datasets are xarray Datasets.

    Example:
        create_exceedance_single_point_dict(toe_dataset, timeseries_dataset)
    """
    
    # Extract the year from toe_ds values
    year = toe_ds.values
    
    # Find the datetime corresponding to the extracted year in timeseries_ds
    year_datetime = timeseries_ds.sel(time=timeseries_ds.time.dt.year==int(year)).time.values[0]
    
    # Find the value corresponding to the extracted year in timeseries_ds
    val = timeseries_ds.sel(time=timeseries_ds.time.dt.year==int(year)).values[0]
    
    # Create and return the dictionary
    return {
        'year': year,
        'year_datetime': year_datetime,
        'val': val
    }





def data_var_pattern_correlation_all_combs(ds: xr.Dataset) -> pd.DataFrame:
    """
    Calculate Spearman correlation between variables in a dataset for all combinations.

    Parameters:
        ds (xr.Dataset): An xarray Dataset containing variables for correlation.

    Returns:
        pd.DataFrame: A DataFrame showing the Spearman correlation between all variables.
                      Rows and columns represent variables, values are correlation coefficients.
                      NaN is placed where the variables are the same or have already been correlated.
    """
    data_vars = ds.data_vars
    tests_used: List[str] = list(data_vars)
    test_combinations = list(combinations(tests_used, 2))  # Generate unique combinations

    # Dictionary to store correlations
    correlations: Dict[str, Dict[str, float]] = {}
    for test_left, test_right in test_combinations:
    
        # Calculate Spearman correlation
        corr_values = spearmanr(
            ds[test_left].values.flatten(),
            ds[test_right].values.flatten(),
            nan_policy='omit'
        ).correlation
        
        correlations.setdefault(test_left, {})[test_right] = corr_values
        # Both combs need to be in there to read into pandas, but the value doesn't need to 
        # be repeated twice - hence nan
        correlations.setdefault(test_right, {})[test_left] = np.nan


    # Convert dictionary to DataFrame
    correlation_df = pd.DataFrame(correlations).round(2)

    # Convert dictionary to DataFrame and reverse column order
    correlation_df = pd.DataFrame(correlations).round(2)
    # Ensuring the row and column order is correct
    correlation_df = correlation_df.loc[data_vars, :][data_vars]
    return correlation_df
    


def calculate_returned_binary_ds(arr: ArrayLike, year_of_emergence: int, time_years: ArrayLike) -> ArrayLike:
    """
    Calculates a binary array representing the emergence of an event.

    Parameters:
        arr (ArrayLike): Input array.
        year_of_emergence (int): Year of the event's emergence.
        time_years (ArrayLike): Array of years.

    Returns:
        ArrayLike: Binary array representing the event's emergence.
                   0 before the year_of_emergence, 1 after.

    If year_of_emergence is NaN or all values in arr are NaN, the function
    returns arr unchanged.
    """

    # If the year_of_emergence is nan or all the values at the location are nan
    # then return nan
    if np.isnan(year_of_emergence) or np.all(np.isnan(arr)): return arr

    # The integer arguement of where the time_years equals the emergence arg
    year_of_emergence = int(year_of_emergence)

    emergence_arg = np.argwhere(time_years == np.round(year_of_emergence))    
    emergence_arg = emergence_arg.item()

    to_return = np.zeros_like(arr)

    # Set all values to 1 after emergence has occured
    to_return[emergence_arg:] =  1
    #print(np.unique(to_return))

    return to_return

def percentage_lat_lons(ds_num, ds_denom, weights):
    """
    Calculate the percentage by comparing two datasets.

    Parameters:
        ds_num (xarray.Dataset): Numerator dataset.
        ds_denom (xarray.Dataset): Denominator dataset.
        weights (xarray.DataArray): Weights for the calculation.

    Returns:
        xarray.DataArray: Dataset containing the percentage.
    """
    # Calculate the sum of the numerator dataset weighted by weights
    ds_num_sum = ds_num.weighted(weights).sum(dim=['lat', 'lon'])

    # Calculate the sum of the denominator dataset weighted by weights
    ds_denom_sum = ds_denom.weighted(weights).sum(dim=['lat', 'lon'])

    # Calculate the percentage
    percent_ds = ds_num_sum / ds_denom_sum * 100

    return percent_ds


def calculate_percent_stable_for_regions(binary_emergence_ds: xr.Dataset, land_mask_ds: xr.Dataset, 
                                         only_1s_ds: xr.Dataset, regions: list, logginglevel='ERROR') -> xr.Dataset:
    """
    Calculate the percentage of stability for regions based on binary emergence data.

    Parameters:
        binary_emergence_ds (xr.Dataset): Dataset containing binary emergence data.
        land_mask_ds (xr.Dataset): Dataset containing land mask data.
        only_1s_ds (xr.Dataset): Dataset with only values of 1 for valid data points.
        regions (list): List of region objects containing a name and latitude slice.
        logginglevel (str): Logging level for the function. Default is 'ERROR'.

    Returns:
        xr.Dataset: Dataset with percentage of stability for each region.
    """
    utils.change_logginglevel(logginglevel)

    weights = compute_weights(binary_emergence_ds)
    ds_collection = []
    points_in_region_dict = {}

    for region in regions:
        region_name, lat_slice = region['name'].lower(), region['slice']

        ds_region, only_1s_ds_region, max_points = prepare_region_datasets(
            binary_emergence_ds, only_1s_ds, land_mask_ds, region_name, lat_slice
        )

        time_series_ds = percentage_lat_lons(ds_region, only_1s_ds_region, weights)
        ds_collection.append(time_series_ds)

        points_in_region_dict[region_name] = compute_region_metadata(
            only_1s_ds_region, max_points
        )

    emergence_time_series_ds = xr.concat(ds_collection, dim='region')
    emergence_time_series_ds.attrs = points_in_region_dict
    
    return emergence_time_series_ds


def compute_weights(ds: xr.Dataset) -> xr.DataArray:
    """Compute weights based on latitude for area-weighted calculations."""
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = 'weights'
    return weights


def prepare_region_datasets(binary_emergence_ds: xr.Dataset, only_1s_ds: xr.Dataset,
                             land_mask_ds: xr.Dataset, region_name: str, lat_slice: slice) -> tuple:
    """Prepare datasets for a specific region, applying masking if needed."""
    ds_region = binary_emergence_ds.sel(lat=lat_slice).expand_dims({'region': [region_name]})
    only_1s_ds_region = only_1s_ds.sel(lat=lat_slice)

    max_points = xr.where(only_1s_ds_region, 1, 1)

    if region_name in ['land', 'ocean']:
        mask_to_use_ds = xr.where(land_mask_ds, 0, 1) if region_name == 'land' else xr.where(land_mask_ds, 1, 0)
        ds_region = ds_region.where(mask_to_use_ds)
        only_1s_ds_region = only_1s_ds_region.where(mask_to_use_ds)
        max_points = max_points.where(mask_to_use_ds)

    return ds_region, only_1s_ds_region, max_points


def compute_region_metadata(only_1s_ds_region: xr.Dataset, max_points: xr.Dataset) -> dict:
    """Compute metadata for a region, including available and maximum points."""
    number_of_points = only_1s_ds_region.sum(dim=['lat', 'lon']).values.item()
    max_number_of_points = max_points.sum(dim=['lat', 'lon']).values.item()
    return {
        'available': number_of_points,
        'maximum': max_number_of_points
    }






# def calculate_percent_stable_for_regions(binary_emergence_ds: xr.Dataset, land_mask_ds:xr.Dataset, only_1s_ds:xr.Dataset,
#                                          logginglevel='ERROR'
#                                         ) -> xr.Dataset:
#     """
#     Calculate the percentage of stability for regions based on binary emergence data.

#     Parameters:
#         binary_emergence_ds (xr.Dataset): Dataset containing binary emergence data.
#         land_mask_ds (xr.Dataset): Dataset containing land mask data.
#         only_1s_ds (xr.Dataset): Dataset with only values of 1. This is needed as it can differ
#                                 due to data availability

#     Returns:
#         xr.Dataset: Dataset with percentage of stability for each region.

#     This function calculates the percentage of stability for regions based on binary
#     emergence data and land mask information. It assumes the existence of a variable
#     'only_1s_ds' that is the same shape as 'binary_emergence_ds' and contains only
#     values of 1 where the event has occurred.
#     """

#     utils.change_logginglevel(logginglevel)
#     # Calculating the weights
#     weights = np.cos(np.deg2rad(binary_emergence_ds.lat))
#     weights.name = 'weights'

#     # The two regions that need the land-sea make
#     NEEDS_MASKING = ['land', 'ocean']

#     ds_collection = [] # Storing all the datasets
#     points_in_region_dict = {} # The number of 
#     for region in toe_const.regionLatLonTuples:
#         region_name = region.value.name.lower()
#         lat_slice = region.value.slice
        
#         # Select data for the current region
#         ds_region = binary_emergence_ds.sel(lat=lat_slice).expand_dims({'region':[region_name]})
#         only_1s_ds_region = only_1s_ds.sel(lat=lat_slice)

#         logger.info(region_name)
#         logger.info(lat_slice)
#         logger.debug(ds_region.lat.values)
#         logger.debug(only_1s_ds_region.lat.values)

#         # Defininf this here so it can be masked if needed
#         max_number_of_points_in_region = xr.where(only_1s_ds_region, 1, 1)
#         # Apply masking if needed
#         if region_name in NEEDS_MASKING:
#             mask_to_use_ds = xr.where(land_mask_ds, 0, 1) if region_name == 'land' else xr.where(land_mask_ds, 1, 0) 
#             ds_region = ds_region.where(mask_to_use_ds)
#             only_1s_ds_region = only_1s_ds_region.where(mask_to_use_ds)
#             max_number_of_points_in_region = max_number_of_points_in_region.where(mask_to_use_ds)

#         # Compute the fraction stable
#         time_series_ds = percentage_lat_lons(ds_region, only_1s_ds_region, weights)

#         ds_collection.append(time_series_ds)

#         # Meta data on how may available point in this region
#         number_of_points_in_region = only_1s_ds_region.sum(dim=['lat', 'lon']).values.item()
#         max_number_of_points_in_region = max_number_of_points_in_region.sum(dim=['lat', 'lon']).values.item()
#         points_in_region_dict[region_name] = {
#             'available': number_of_points_in_region, 
#             'maximimum':max_number_of_points_in_region}
        
#     emergence_time_series_ds = xr.concat(ds_collection, dim='region')
#     emergence_time_series_ds.attrs = points_in_region_dict
#     logger.info('\n')
#     return emergence_time_series_ds


def find_value_at_emergence_arg(arr: ArrayLike, year_of_emergence: int, time_years: ArrayLike) -> float:
    """
    Finds the value in the `arr` array at the index corresponding to the `year_of_emergence` in `time_years`.

    Parameters:
        arr (ArrayLike): The array of values.
        year_of_emergence (int): The year of emergence to find the value for.
        time_years (ArrayLike): The array of years.

    Returns:
        float: The value in `arr` corresponding to the `year_of_emergence`, or NaN if `year_of_emergence` is NaN
    """
    # If year_of_emergence is NaN, return NaN
    if np.isnan(year_of_emergence):
        return np.nan
    
    # Find the index of year_of_emergence in time_years
    emergence_arg = np.argwhere(time_years == year_of_emergence).item()
    
    # Get the value in arr at the emergence_arg index
    value_at_arg = arr[emergence_arg]
    
    return value_at_arg
