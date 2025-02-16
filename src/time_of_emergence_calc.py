import os, sys
from functools import partial
from itertools import groupby
from typing import Optional, Callable, Union, Tuple, NamedTuple, List
from itertools import combinations
from enum import Enum


import numpy as np
import pandas as pd
import xarray as xr
import dask.array as daskarray
from scipy.stats import anderson_ksamp, ks_2samp,ttest_ind, spearmanr, levene, mannwhitneyu, kruskal, gaussian_kde

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
return_kruskal_pvalue = partial(return_statistical_pvalue, stats_test=kruskal)





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



def calculate_freedman_diaconis_bins(arr=None, length=None, logginglevel="ERROR"):
    """
    Calculate bin edges using the Freedman-Diaconis rule for a 1D NumPy array.
    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    Parameters:
    - arr: numpy.ndarray
        A 1D NumPy array containing all data points.

    Returns:
    - bin_edges: numpy.ndarray
        The bin edges calculated using the Freedman-Diaconis rule.
    """
    utils.change_logginglevel(logginglevel)
    # Remove NaNs
    arr = arr[~np.isnan(arr)]
        

    # If passing in girdded data
    if length is None: length = len(arr)
    
    # Calculate bin width using the Freedman-Diaconis rule
    p75 = np.percentile(arr, 75)
    p25 = np.percentile(arr, 25)
    iqr =  p75 - p25
    print(f'{p75=}, {p25=}, {iqr=}, {length=}')
    
    bin_width = 2 * iqr / length ** (1 / 3)
    logger.info(bin_width)

    # Define bin edges
    bin_edges = np.arange(np.nanmin(arr)-bin_width, np.nanmax(arr) + bin_width, bin_width)

    return bin_edges
    
def rel_freq(arr:np.ndarray, bins:np.ndarray)->np.ndarray:
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


get_rel_freq = rel_freq

def discrete_pdf(arr, bins:np.ndarray=None, num_bins:int=None) -> np.ndarray:
    """
    Computes the discrete probability density function (PDF) of the input array.

    Parameters:
    -----------
    arr : np.ndarray
        Input array of data points.
    bins : np.ndarray, optional
        An array of bin edges. If not provided, bins will be automatically generated.
    num_bins : int, optional
        The number of bins to use if bins are not provided. Default is 25.

    Returns:
    --------
    bins : np.ndarray
        The array of bin edges used for computing the PDF.
    rel_freq : np.ndarray
        The relative frequency of data points within each bin, representing the PDF.
    """
    if num_bins: print('num_bins is deprecated - please remove')
    
    if bins is None: bins = calculate_freedman_diaconis_bins(arr)#np.linspace(np.nanmin(arr), np.nanmax(arr), num_bins)

    rel_freq = get_rel_freq(arr, bins)

    return bins, rel_freq


def discrete_distribution_overlap(rel_freq_base, rel_freq_arr):
    # Stack the relative frequencies for comparison
    freq_stack = np.vstack([rel_freq_base, rel_freq_arr])
    
    # Find the minimum relative frequency at each bin
    freq_min = np.nanmin(freq_stack, axis=0)
    
    # Sum the minimum frequencies and convert to a percentage
    freq_min_sum_percent = np.sum(freq_min) * 100

    return freq_min_sum_percent


def perkins_skill_score(arr:np.ndarray, base_arr:np.ndarray, bins:np.ndarray=None)->float:
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

    if bins is None:

        bins = calculate_freedman_diaconis_bins(
            np.concatenate(np.concatenate([base_arr, arr]), length=len(base_arr))
        )
    
    # Calculate the relative frequencies for each array
    rel_freq_base = get_rel_freq(base_arr, bins)
    rel_freq_arr = get_rel_freq(arr, bins)
    
    overlap = discrete_distribution_overlap(rel_freq_base, rel_freq_arr)
    
    return float(overlap)



def perkins_skill_score_base_bins(arr:np.ndarray, rel_freq_base:np.ndarray, bins:np.ndarray=None)->float:
    """
    Calculate the Perkins Skill Score (PSS) between two arrays.
    
    Parameters:
    arr_best (numpy.ndarray): First input array of values.
    rel_freq_base (numpy.ndarray): relative frequency of base period.
    
    Returns:
    float: Perkins Skill Score as a percentage. Returns NaN if any array is fully NaN.
    """
    # Check if any input array is fully NaN
    if np.all(np.isnan(arr)) or np.all(np.isnan(rel_freq_base)): return np.nan


    # Calculate the relative frequencies for each array
    rel_freq_arr = get_rel_freq(arr, bins)
    
    overlap = discrete_distribution_overlap(rel_freq_base, rel_freq_arr)
    
    return float(overlap)



def calculate_hellinger_distance(
    dist1: np.ndarray, 
    dist2: np.ndarray, 
    x: np.ndarray
) -> float:
    """
    Calculate the Hellinger distance between two probability distributions.

    The Hellinger distance measures the similarity between two probability
    distributions and is bounded between 0 (identical distributions) and 1
    (maximally different distributions).

    Args:
        dist1: An ndarray of values representing the first probability distribution.
        dist2: An ndarray of values representing the second probability distribution.
        x: An ndarray of evenly spaced points where the distributions are evaluated.

    Returns:
        The Hellinger distance between the two distributions.
    """
    # Compute the pointwise squared difference between the square roots of the distributions
    squared_diff = (np.sqrt(dist1) - np.sqrt(dist2 ))**2

    hellinger_distance = np.sqrt(np.sum(squared_diff) * (x[1] - x[0])) / np.sqrt(2)

    return hellinger_distance * 100

def create_x(arr:np.ndarray=None, bmin:float=None, bmax:float=None, num_points=1000) -> np.ndarray:
    
    if bmin is None or bmax is None:
        bmin = np.nanmin(arr)
        bmax = np.nanmax(arr)
        
    val_range = bmax-bmin
   # Extend the range a bit
    bmax = bmax+val_range/3
    bmin = bmin-val_range/3
    
    x = np.linspace(bmin, bmax, num_points)
    return x

# def create_kde(arr: np.ndarray, x:np.ndarray=None, bmin:float=None, bmax:float=None, **kwargs):
    # ''''''

    # # No x values provided - created own
    # if x is None: x = create_x(arr, bmin, bmax)
        
    # # Remove NaN and infinite values from the arrays
    # arr = arr[np.isfinite(arr)]    
    # # Compute the KDE for each array
    # kde = gaussian_kde(arr, **kwargs)
    # kde_vals = kde(x)

    # kde_vals /= np.trapz(kde_vals, x)

    # return x, kde_vals

def create_kde(arr: np.ndarray, x:np.ndarray=None, bmin:float=None, bmax:float=None, **kwargs):
    """
    Computes the Kernel Density Estimate (KDE) of the input array using Gaussian KDE.

    Parameters
    ----------
    arr : np.ndarray
        Input array for which the KDE is computed. NaN and infinite values are removed.
    x : np.ndarray, optional
        Array of x values at which to evaluate the KDE. If not provided, 
        t is generated using `create_x(arr, bmin, bmax)`.
    bmin : float, optional
        Minimum boundary for x values when `x` is not provided.
    bmax : float, optional
        Maximum boundary for x values when `x` is not provided.
    **kwargs : dict
        Additional keyword arguments passed to `gaussian_kde`.

    Returns
    -------
    x : np.ndarray
        The x values at which the KDE is evaluated.
    kde_vals : np.ndarray
        The computed KDE values, normalized so that the integral over `x` equals 1.

    Notes
    -----
    - The function automatically removes NaN and infinite values from `arr` before computing the KDE.
    - The KDE is computed using `scipy.stats.gaussian_kde` and normalized using the trapezoidal rule (`np.trapz`).
    """
    # No x values provided - created own
    if x is None: x = create_x(arr, bmin, bmax)
        
    # Remove NaN and infinite values from the arrays
    arr = arr[np.isfinite(arr)]    
    # Compute the KDE for each array
    kde = gaussian_kde(arr, **kwargs)
    kde_vals = kde(x)

    kde_vals /= np.trapz(kde_vals, x)

    return x, kde_vals



def calculate_kde_overlap(dist1, dist2, x):
    # Calculate the overlap shape by taking the minimum of the two KDEs at each point
    overlap_shape = np.min(np.vstack([dist1, dist2]), axis=0)
    
    # Integrate the overlap shape to find the overlap area
    overlap_area = np.trapz(overlap_shape, x)
    
    # Convert the overlap area to a percentage
    overlap_percent = overlap_area * 100

    return overlap_percent




def caculate_distribution_overlap(*args, **kwargs):
    print('Function now called calculate_kde_overlap')
    return calculate_kde_overlap(*args, **kwargs)


def __overlap_helper_function(arr_future: np.ndarray, arr_base: np.ndarray, return_all=False,
                             method_kwargs=None, bmax=None, bmin=None, overlap_function=None) -> float:
    """
    Helper function to calculate the overlap between the KDEs of two arrays using a specified overlap function.

    Parameters:
    arr_future (numpy.ndarray): First input array of values.
    arr_base (numpy.ndarray): Second input array of values.
    return_all (bool): If False (default) just return the overlap percent. If True,
                        return the KDEs and the overlap percent.
    kde_kwargs (dict, optional): Keyword arguments to pass to the KDE creation function.
    bmax (float, optional): Maximum value for the range of the KDE.
    bmin (float, optional): Minimum value for the range of the KDE.
    overlap_function (callable, optional): Function to calculate overlap between two distributions.
                                           Should accept `kde_base`, `kde_future`, and `x` as arguments.

    Returns:
    float: Overlap area as calculated by the specified overlap function. Returns NaN if any array is fully NaN.
    """

    if not method_kwargs:
        method_kwargs = {}

    # Check if any input array is fully NaN
    if np.all(np.isnan(arr_future)) or np.all(np.isnan(arr_base)):
        return np.nan

    # Find the maximum and minimum values from the combined arrays
    if bmax is None:
        bmax = np.nanmax(np.concatenate([arr_base, arr_future]))
    if bmin is None:
        bmin = np.nanmin(np.concatenate([arr_base, arr_future]))

    x = create_x(bmin=bmin, bmax=bmax)

    _, kde_base = create_kde(arr_base, x, **method_kwargs)
    _, kde_future = create_kde(arr_future, x, **method_kwargs)

    if overlap_function is None:
        raise ValueError("An overlap function must be provided.")

    out_metric = overlap_function(kde_base, kde_future, x)

    if return_all:
        return x, kde_base, kde_future, out_metric

    return out_metric



fractional_geometric_area = partial(__overlap_helper_function, overlap_function=calculate_kde_overlap)
hellinger_distance = partial(__overlap_helper_function, overlap_function=calculate_hellinger_distance)


def farctional_geometric_area(*arg, **kwargs):
    print('THis is the typo function - it has been fixed and now is fractional_geometric_area')
    return fractional_geometric_area(*args, **kwargs)



def __overlap_helper_function_base_fitted(arr_future: np.ndarray, kde_base: np.ndarray, overlap_function, x) -> float:
    """
    Helper function to calculate the overlap between the KDEs of two arrays using a specified overlap function.

    Parameters:
    arr_future (numpy.ndarray): First input array of values.
    arr_base (numpy.ndarray): Second input array of values.
    return_all (bool): If False (default) just return the overlap percent. If True,
                        return the KDEs and the overlap percent.
    kde_kwargs (dict, optional): Keyword arguments to pass to the KDE creation function.
    bmax (float, optional): Maximum value for the range of the KDE.
    bmin (float, optional): Minimum value for the range of the KDE.
    overlap_function (callable, optional): Function to calculate overlap between two distributions.
                                           Should accept `kde_base`, `kde_future`, and `x` as arguments.

    Returns:
    float: Overlap area as calculated by the specified overlap function. Returns NaN if any array is fully NaN.
    """

    if not method_kwargs: method_kwargs = {}

    # Check if any input array is fully NaN
    if np.all(np.isnan(arr_future)) or np.all(np.isnan(arr_base)): return np.nan


    _, kde_future = create_kde(arr_future, x, **method_kwargs)

    out_metric = overlap_function(kde_base, kde_future, x)

    return out_metric

    
    




##------------------------- Function related to getting ToE from metric

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
    if comparison_func is not None: # Can be none if values are already bool
        greater_than_arg_list = comparison_func(arr, threshold)
    else:
        greater_than_arg_list = arr

    # If no value exceeds threshold, return nan
    if np.all(greater_than_arg_list == False): return np.nan

    # Group consecutive True and False values
    groups = [(key, len(list(group))) for key, group in groupby(greater_than_arg_list)]

    # If the last group is False, there is no exceedance, return nan
    if groups[-1][0] == False: return np.nan

    # The argument will be the sum of all the other group lengths up to the last group
    # As the -1 group is being used, this will be when permanent emergence occurs
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
    if time is None: time = ds.time.dt.year.values
        
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
    to_retrun = xr.apply_ufunc(
        partial_exceedance_func, 
        ds, 
        **exceedance_dict
    )

    return to_retrun




