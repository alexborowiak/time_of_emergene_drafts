import os, sys
from functools import partial
from itertools import groupby
from typing import Optional, Callable, Union, Tuple, NamedTuple
from itertools import combinations
from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as daskarray
from scipy.stats import anderson_ksamp, ks_2samp,ttest_ind, spearmanr
from numpy.typing import ArrayLike

# My imports
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts'))
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))

import toe_constants as toe_const
import utils
logger = utils.get_notebook_logger()

def return_ttest_pvalue(test_arr, base_arr):
    """
    Compute T-Test p-value between two arrays.

    Parameters:
        test_arr (ArrayLike): Array to test against base_arr.
        base_arr (ArrayLike): Base array to compare against.

    Returns:
        float: T-Test p-value.
    """
    return ttest_ind(test_arr, base_arr, nan_policy='omit').pvalue

def return_ks_pvalue(test_arr, base_arr):
    """
    Compute Kolmogorov-Smirnov test p-value between two arrays.

    Parameters:
        test_arr (ArrayLike): Array to test against base_arr.
        base_arr (ArrayLike): Base array to compare against.

    Returns:
        float: Kolmogorov-Smirnov test p-value.
    """
    return ks_2samp(test_arr, base_arr).pvalue


def return_anderson_pvalue(test_arr, base_arr):
    """
    Compute Anderson-Darling test p-value between two arrays.

    Parameters:
        test_arr (ArrayLike): Array to test against base_arr.
        base_arr (ArrayLike): Base array to compare against.

    Returns:
        float: Anderson-Darling test p-value.
    """
    if all(np.isnan(test_arr)) or all(np.isnan(base_arr)): return np.nan
    # print(test_arr.shape, base_arr.shape)
    return anderson_ksamp([test_arr, base_arr]).pvalue



def stats_test_1d_array(arr, stats_func:Callable, window: int=20, base_period_length:int = 50):
    """
    Apply stats_func test along a 1D array.

    Parameters:
        arr (ArrayLike): 1D array to apply the test to.
        window (int): Size of the rolling window for the test.
        base_period_length (int, optional): Length of the base period. Defaults to 50.

    Returns:
        ArrayLike: Array of p-values.
    """
    # The data to use for the base period
    base_list = arr[:base_period_length]
    # Stop when there are not enough points left
    number_iterations = arr.shape[0] - window
    pval_array = np.zeros(number_iterations)
    
    for t in np.arange(number_iterations):
        arr_subset = arr[t: t+window]
        p_value = stats_func(base_list, arr_subset) # return_ttest_pvalue
        pval_array[t] = p_value

    # TODO: This could be done in the apply_ufunc
    lenghth_diff = arr.shape[0] - pval_array.shape[0]
    pval_array = np.append(pval_array, np.array([np.nan] *lenghth_diff))
    return pval_array 


def stats_test_1d_xr(ds: xr.Dataset, stats_func: Callable, window: int = 20, base_period_length: int = 50) -> ArrayLike:
    """
    Calculate statistical test for 1D data along the specified dimension using a custom function.

    Parameters:
    - ds (xr.Dataset): The xarray Dataset containing the data.
    - stats_func (Callable): The statistical function to apply to the data.
    - window (int): Size of the rolling window for calculations. Default is 20.
    - base_period_length (int): Length of the base period for the statistical test. Default is 50.

    Returns:
    - ArrayLike: An xarray DataArray containing the statistical test results.

    This function calculates a statistical test for 1D data along the specified dimension of an xarray Dataset.
    It applies a custom statistical function (stats_func) to the data using a rolling window of size 'window'.
    The base period length for the statistical test is 'base_period_length'.
    The result is returned as an xarray DataArray with the same dimensions as the input Dataset.
    """
    # Call the underlying function that operates on NumPy arrays
    arr = stats_test_1d_array(ds, stats_func=stats_func, window=window, base_period_length=base_period_length)

    # Remove the nan-values
    arr = arr[np.isfinite(arr)]

    # We want to use the mid point. However, this calculation uses the start point
    half_window = int(window/2)
    # Create a new xarray DataArray filled with the calculated values
    result = xr.zeros_like(ds.isel(time=slice(half_window, half_window+len(arr)))) + arr
    
    return result

def return_hawkins_signal_and_noise(lt: ArrayLike, gt: ArrayLike, return_reconstruction:bool=False) -> Tuple[ArrayLike, ArrayLike]:
    """
    Calculate the signal and noise using the Hawkins method.

    Parameters:
        lt (ArrayLike): Time series data to be filtered.
        gt (ArrayLike): Time series used as the reference for filtering.
        return_reconstruction (Tuple) = False:
            Returns the reconstruction of the local time series.
            This is optional, as the reconstruction series is only needed for verification purposes                                   

    Returns:
        Tuple[ArrayLike, ArrayLike]: A tuple containing the filtered signal and noise.

    If either `lt` or `gt` contains all NaN values, it returns `lt` as both the signal and noise.

    The Hawkins method removes NaNs from the start and end of `lt` and `gt` to align the series.
    It then calculates the gradient `grad` and y-intercept `yint` of the linear fit between `gt` and `lt`.
    The signal is calculated as `signal = grad * gt`.
    The noise is calculated as the difference between `lt` and `signal`.

    NaN values are padded back to the filtered signal and noise arrays to match the original input length.
    """

    if np.all(np.isnan(lt)) or np.all(np.isnan(gt)):
        # If either series is all NaN, return lt as both signal and noise
        return lt, lt

    # If either is nan we want to drop
    nan_locs = np.isnan(lt)#  | np.isnan(gt)

    lt_no_nan = lt[~nan_locs]
    gt_no_nan = gt[~nan_locs]

    # Calculate the gradient and y-intercept
    grad, yint = np.polyfit(gt_no_nan, lt_no_nan, deg=1)

    # Calculate signal and noise
    signal = grad * gt_no_nan
    noise = lt_no_nan - signal

    signal_to_return = np.empty_like(gt)
    noise_to_return = np.empty_like(lt)
    
    signal_to_return.fill(np.nan)
    noise_to_return.fill(np.nan)

    signal_to_return[~nan_locs] = signal
    noise_to_return[~nan_locs] = noise

    
    if return_reconstruction:
        reconstructed_lt = grad * gt + yint
        return signal_to_return, noise_to_return, reconstructed_lt
    return signal_to_return, noise_to_return

    # # Pad NaNs back to the filtered signal and noise arrays
    # signal = np.concatenate([[np.nan] * number_nans_at_start, signal, [np.nan] * number_nans_at_end])
    # noise = np.concatenate([[np.nan] * number_nans_at_start, noise, [np.nan] * number_nans_at_end])

    # # Find the number of NaNs at the start and end of lt
    # number_nans_at_start = np.where(~np.isnan(lt))[0][0]
    # number_nans_at_end = np.where(~np.isnan(lt[::-1]))[0][0]

    # # Remove start NaNs
    # lt = lt[number_nans_at_start:]
    # gt = gt[number_nans_at_start:]

    # # Remove end NaNs if there are any
    # if number_nans_at_end > 0:
    #     lt = lt[:-number_nans_at_end]
    #     gt = gt[:-number_nans_at_end]

def get_exceedance_arg(arr, time, threshold, comparison_func):
    """
    Get the index of the first occurrence where arr exceeds a threshold.

    Parameters:
        arr (array-like): 1D array of values.
        time (array-like): Corresponding 1D array of time values.
        threshold (float): Threshold value for comparison.
        comparison_func (function): Function to compare arr with the threshold.

    Returns:
        float: The time corresponding to the first exceedance of the threshold.
               If there is no exceedance, returns np.nan.

    Example:
        data = [False, False, False, False, False, False,
                False, False, False, False, True, False, True, 
                True, True]

        # Group consecutive True and False values
        groups = [(key, len(list(group))) for key, group in groupby(data)]
        print(groups)
        >>> [(False, 10), (True, 1), (False, 1), (True, 3)]
        # Check if the last group is True
        groups[-1][0] == True
        # Compute the index of the first exceedance
        first_exceedance_arg = int(np.sum(list(map(lambda x: x[1], groups))[:-1]))
        print(first_exceedance_arg)
        >>> 12
    """
    # Entire nan slice, return nan
    if all(np.isnan(arr)):
        return np.nan

    # Find indices where values exceed threshold
    greater_than_arg_list = comparison_func(arr, threshold)

    # If no value exceeds threshold, return nan
    if np.all(greater_than_arg_list == False):
        return np.nan

    # Group consecutive True and False values
    groups = [(key, len(list(group))) for key, group in groupby(greater_than_arg_list)]

    # If the last group is False, there is no exceedance, return nan
    if groups[-1][0] == False:
        return np.nan

    # The argument will be the sum of all the other group lengths up to the last group
    first_exceedance_arg = int(np.sum(list(map(lambda x: x[1], groups))[:-1]))

    # Get the time corresponding to the first exceedance
    first_exceedance = time[first_exceedance_arg]

    return first_exceedance

def get_permanent_exceedance(ds: xr.DataArray, threshold: Union[int, float], comparison_func: Callable,
                             time: Optional[xr.DataArray] = None)-> xr.DataArray:
    """
    Calculate the time of the first permanent exceedance for each point in a DataArray.

    This function calculates the time of the first permanent exceedance (defined as when a value exceeds a threshold
    and never goes below it again) for each point in a DataArray.

    Parameters:
        ds (xr.DataArray): Input data.
        threshold (Union[int, float]): Threshold value for exceedance.
        comparison_func (Callable): Function to compare values with the threshold.
        time (Optional[xr.DataArray]): Optional array of time values corresponding to the data. 
                                        If not provided, it will default to the 'year' component of ds's time.

    Returns:
        xr.DataArray: DataArray containing the time of the first permanent exceedance for each point.
    """
    # If time is not provided, use 'year' component of ds's time
    if time is None:
        time = ds.time.dt.year.values
        
    # Partial function to compute the exceedance argument
    partial_exceedance_func = partial(get_exceedance_arg, time=time, threshold=threshold, comparison_func=comparison_func)
               
    # Dictionary of options for xr.apply_ufunc
    exceedance_dict = dict(
        input_core_dims=[['time']],
        output_core_dims=[[]],
        vectorize=True, 
        dask='parallelized',
        output_dtypes=[float],
        keep_attrs='identical'
    )

    # Apply the partial function to compute the permanent exceedance
    return xr.apply_ufunc(
        partial_exceedance_func, 
        ds, 
        **exceedance_dict
    )

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

    emergence_arg = np.argwhere(time_years == np.round(year_of_emergence)).item()

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



def calculate_percent_stable_for_regions(binary_emergence_ds: xr.Dataset, land_mask_ds:xr.Dataset, only_1s_ds:xr.Dataset,
                                         logginglevel='ERROR'
                                        ) -> xr.Dataset:
    """
    Calculate the percentage of stability for regions based on binary emergence data.

    Parameters:
        binary_emergence_ds (xr.Dataset): Dataset containing binary emergence data.
        land_mask_ds (xr.Dataset): Dataset containing land mask data.
        only_1s_ds (xr.Dataset): Dataset with only values of 1. This is needed as it can differ
                                due to data availability

    Returns:
        xr.Dataset: Dataset with percentage of stability for each region.

    This function calculates the percentage of stability for regions based on binary
    emergence data and land mask information. It assumes the existence of a variable
    'only_1s_ds' that is the same shape as 'binary_emergence_ds' and contains only
    values of 1 where the event has occurred.
    """

    utils.change_logginglevel(logginglevel)
    # Calculating the weights
    weights = np.cos(np.deg2rad(binary_emergence_ds.lat))
    weights.name = 'weights'

    # The two regions that need the land-sea make
    NEEDS_MASKING = ['land', 'ocean']

    ds_collection = [] # Storing all the datasets
    points_in_region_dict = {} # The number of 
    for region in toe_const.regionLatLonTuples:
        region_name = region.value.name.lower()
        lat_slice = region.value.slice
        
        # Select data for the current region
        ds_region = binary_emergence_ds.sel(lat=lat_slice).expand_dims({'region':[region_name]})
        only_1s_ds_region = only_1s_ds.sel(lat=lat_slice)

        logger.info(region_name)
        logger.info(lat_slice)
        logger.debug(ds_region.lat.values)
        logger.debug(only_1s_ds_region.lat.values)

        # Defininf this here so it can be masked if needed
        max_number_of_points_in_region = xr.where(only_1s_ds_region, 1, 1)
        # Apply masking if needed
        if region_name in NEEDS_MASKING:
            mask_to_use_ds = xr.where(land_mask_ds, 0, 1) if region_name == 'land' else xr.where(land_mask_ds, 1, 0) 
            ds_region = ds_region.where(mask_to_use_ds)
            only_1s_ds_region = only_1s_ds_region.where(mask_to_use_ds)
            max_number_of_points_in_region = max_number_of_points_in_region.where(mask_to_use_ds)

        # Compute the fraction stable
        time_series_ds = percentage_lat_lons(ds_region, only_1s_ds_region, weights)

        ds_collection.append(time_series_ds)

        # Meta data on how may available point in this region
        number_of_points_in_region = only_1s_ds_region.sum(dim=['lat', 'lon']).values.item()
        max_number_of_points_in_region = max_number_of_points_in_region.sum(dim=['lat', 'lon']).values.item()
        points_in_region_dict[region_name] = {
            'available': number_of_points_in_region, 
            'maximimum':max_number_of_points_in_region}
        
    emergence_time_series_ds = xr.concat(ds_collection, dim='region')
    emergence_time_series_ds.attrs = points_in_region_dict
    logger.info('\n')
    return emergence_time_series_ds


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