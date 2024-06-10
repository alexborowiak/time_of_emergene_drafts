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
from scipy.stats import anderson_ksamp, ks_2samp,ttest_ind, spearmanr, levene, mannwhitneyu
from numpy.typing import ArrayLike

# My imports
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts'))
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))

import toe_constants as toe_const
import utils
logger = utils.get_notebook_logger()




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
    # This change from pvalue to significance_level for some reasons
    try: 
        pval = anderson_ksamp([test_arr, base_arr]).significance_level
    except ValueError:
        return np.nan
    return pval


def remove_nans(arr1, arr2):
    """
    Remove NaN values from two input arrays.

    Parameters:
    arr1 (numpy array): The first input array.
    arr2 (numpy array): The second input array.

    Returns:
    tuple: Two arrays with NaN values removed.

    Notes:
    This function uses NumPy's isfinite function to identify and remove NaN values.
    """
    # Remove NaN values using NumPy's isfinite function
    arr1 = arr1[np.isfinite(arr1)]  # Remove NaN values from arr1
    arr2 = arr2[np.isfinite(arr2)]  # Remove NaN values from arr2

    return arr1, arr2  # Return the updated arrays

def return_statistical_pvalue(arr1, arr2, stats_test):
    """
    Calculate the p-value for a given statistical test for two arrays.

    Parameters:
    arr1 (numpy array): The first input array.
    arr2 (numpy array): The second input array.

    Returns:
    float: The p-value of the specified statistical test.

    Notes:
    If either array contains only NaN values, the function returns NaN.
    """
    # Check if all values are nan
    if np.all(np.isnan(arr1)) or np.all(np.isnan(arr2)): return np.nan
    arr1, ar2 = remove_nans(arr1, arr2)

    return stats_test(arr1, arr2).pvalue


# Initialising multiple p-value tests
return_ttest_pvalue = partial(return_statistical_pvalue, stats_test=ttest_ind)
return_ks_pvalue = partial(return_statistical_pvalue, stats_test=ks_2samp)
return_levene_pvalue = partial(return_statistical_pvalue, stats_test=levene)
return_mannwhitney_pvalue = partial(return_statistical_pvalue, stats_test=mannwhitneyu)



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

    # Re-add all the nans back in where they were before
    signal_to_return.fill(np.nan)
    noise_to_return.fill(np.nan)

    signal_to_return[~nan_locs] = signal
    noise_to_return[~nan_locs] = noise

    
    if return_reconstruction:
        reconstructed_lt = grad * gt + yint
        return signal_to_return, noise_to_return, reconstructed_lt
    return signal_to_return, noise_to_return


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
    # print(year_of_emergence)
    emergence_arg = np.argwhere(time_years == np.round(year_of_emergence))    
    #print(year_of_emergence, emergence_arg)   
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



import numpy as np
from scipy.stats import gaussian_kde

def get_rel_freq(arr:np.ndarray, bins:np.ndarray)->np.ndarray:
    """
    Calculate the relative frequencies of values in the array within the specified bins.
    
    Parameters:
    arr (numpy.ndarray): Input array of values.
    bins (numpy.ndarray): Array of bin edges.

    Returns:
    numpy.ndarray: Relative frequencies of values within the bins.
    """
    # Remove NaN and infinite values from the array
    arr = arr[np.isfinite(arr)]
    
    # Calculate the counts of values in each bin
    counts, _ = np.histogram(arr, bins=bins, density=False)
    
    # Calculate the relative frequencies
    rel_freq = counts / len(arr)
    return rel_freq

def farctional_geometric_area(arr_best:np.ndarray, base_arr:np.ndarray)->float:
    """
    Calculate the fractional geometric area between the KDEs of two arrays.
    
    Parameters:
    arr_best (numpy.ndarray): First input array of values.
    base_arr (numpy.ndarray): Second input array of values.
    
    Returns:
    float: Fractional geometric overlap area as a percentage. Returns NaN if any array is fully NaN.
    """
    # Check if any input array is fully NaN
    if np.all(np.isnan(arr_best)) or np.all(np.isnan(base_arr)): return np.nan
    
    # Find the maximum and minimum values from the combined arrays
    bmax = np.nanmax(np.concatenate([base_arr, arr_best]))
    bmin = np.nanmin(np.concatenate([base_arr, arr_best]))
    
    # Generate a linear space between the minimum and maximum values
    x = np.linspace(bmin, bmax, 1000)
    
    # Remove NaN and infinite values from the arrays
    base_arr = base_arr[np.isfinite(base_arr)]
    arr_best = arr_best[np.isfinite(arr_best)]
    
    # Compute the KDE for each array
    kde1 = gaussian_kde(base_arr)
    kde2 = gaussian_kde(arr_best)
    kde_vals1 = kde1(x)
    kde_vals2 = kde2(x)
    
    # Calculate the overlap shape by taking the minimum of the two KDEs at each point
    overlap_shape = np.min(np.vstack([kde_vals2, kde_vals1]), axis=0)
    
    # Integrate the overlap shape to find the overlap area
    overlap_area = np.trapz(overlap_shape, x)
    
    # Convert the overlap area to a percentage
    overlap_percent = overlap_area * 100
    return overlap_percent

def perkins_skill_score(arr:np.ndarray, base_arr:np.ndarray)->float:
    """
    Calculate the Perkins Skill Score (PSS) between two arrays.
    
    Parameters:
    arr_best (numpy.ndarray): First input array of values.
    base_arr (numpy.ndarray): Second input array of values.
    
    Returns:
    float: Perkins Skill Score as a percentage. Returns NaN if any array is fully NaN.
    """
    # Check if any input array is fully NaN
    if np.all(np.isnan(arr)) or np.all(np.isnan(base_arr)): return np.nan
    
    # Find the maximum and minimum values from the combined arrays
    bmax = np.nanmax(np.concatenate([base_arr, arr]))
    bmin = np.nanmin(np.concatenate([base_arr, arr]))
    
    # Define the bin width and create bin edges
    bins = np.linspace(bmin, bmax, 25)
    
    #step = 0.5
    #bins = np.arange(bmin, bmax + step, step)
    
    # Calculate the relative frequencies for each array
    rel_freq_base = get_rel_freq(base_arr, bins)
    rel_freq_arr = get_rel_freq(arr, bins)
    
    # Stack the relative frequencies for comparison
    freq_stack = np.vstack([rel_freq_base, rel_freq_arr])
    
    # Find the minimum relative frequency at each bin
    freq_min = np.nanmin(freq_stack, axis=0)
    
    # Sum the minimum frequencies and convert to a percentage
    freq_min_sum_percent = np.sum(freq_min) * 100
    
    return float(freq_min_sum_percent)
