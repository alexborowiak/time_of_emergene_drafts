import numpy as np
import xarray as xr
from functools import partial

import utils

import statsmodels.api as sm 
lowess = sm.nonparametric.lowess

from typing import Literal
from numpy.typing import ArrayLike
from typing import Optional, Dict, Callable


logger = utils.get_notebook_logger()

from enum import Enum


def dask_percentile(array: np.ndarray, axis: str, q: float):
    '''
    Applies np.percetnile in dask across an axis
    Parameters:
    -----------
    array: the data to apply along
    axis: the dimension to be applied along
    q: the percentile
    
    Returns:
    --------
    qth percentile of array along axis
    
    Example
    -------
    xr.Dataset.data.reduce(xca.dask_percentile,dim='time', q=90)
    '''
    return array.map_blocks(
        np.nanpercentile,
        axis=axis,
        q=q,
        dtype=array.dtype,
        drop_axis=axis)

class detrendingMethods(str, Enum):
    POLYNOMIAL = 'polynomial'
    LOWESS = 'lowess'
    
def polynomial_fit(y: ArrayLike, x:Optional[ArrayLike] = None, order:float=None, deg:float=None, 
                  nan_removal: Literal['all', 'endpoints'] = 'endpoints') -> ArrayLike:
    """
    Perform a polynomial fit for line y using the Vandermonde matrix method.
    
    Args:
        y (ArrayLike): y-values to be used for fitting.
        x (Optional[ArrayLike]): Optional x-values that can be used. These values are only needed if they are
            not linearly increasing.
        order/deg (float): The order of the polynomial.
    
    Returns:
        ArrayLike: The fitted line.
    """
    if all(np.isnan(y)): return y # All values are nan, don't proceed


    if nan_removal == 'endpoints':
        # First need to deal with any nan values at the start or the end
        number_nans_at_start = np.where(~np.isnan(y))[0][0]
        number_nans_at_end = np.where(~np.isnan(y[::-1]))[0][0]
        # Remove these nans
        y = y[number_nans_at_start:] # Remove start nans
        if number_nans_at_end > 0:  y = y[:-number_nans_at_end] # If number_nans_at_end is then this removes all values
        # If x is not provided, generate linearly increasing values
        x = np.arange(len(y)) if x is None else x
        x = x[:len(y)]
    if nan_removal == 'all':
        nan_locs = np.isfinte(y) * np.isfinite(x)
        x = x[nan_locs]
        y = y[nan_locs]
    # Perform polynomial fit using numpy's polyfit function
    deg = order if order is not None else deg
    deg = 1 if deg is None else deg
    coeff = np.polyfit(x, y, deg=deg)
    # Generate the fitted line using the computed coefficients
    fitted_line = np.polyval(coeff, x)

    # Re-add the nans back in to maintain the length
    fitted_line_lenght_maintained = np.concatenate([[np.nan]*number_nans_at_start, fitted_line, [np.nan] *number_nans_at_end])
    return fitted_line_lenght_maintained

def lowess_fit(exog: Callable, window:int=50) -> Callable:
    '''
    A function to fill the lowess function with exog (x values) and the fraction
    '''
    return partial(lowess, exog=exog, frac=window/len(exog), return_sorted=False)


def apply_lowess(arr, window=41):
    """
    Apply the LOWESS (Locally Weighted Scatterplot Smoothing) algorithm to a 1D array.

    Parameters:
    arr (numpy array): The input array to be smoothed.

    Returns:
    numpy array: The smoothed array.

    Notes:
    If the input array contains only NaN values, the function returns the original array.
    """
    # Check if all elements in the array are NaN
    if all(np.isnan(arr)): return arr

    # Create an array of x values (indices) for the input array
    x = np.arange(arr.shape[0])

    # Apply the LOWESS algorithm
    yhat = lowess(arr, x, frac=window/len(arr), return_sorted=False)

    return yhat


def apply_detrend_as_ufunc(
    da: xr.DataArray, func1d: Callable, func_kwargs:Optional[Dict]=None, debug=False) -> xr.DataArray:
    '''
    Applies the detrending funcs as a ufunc.
    '''
    
    if debug: print(func_kwargs)
    if func_kwargs is not None:
        func1d = partial(func1d, **func_kwargs)
        
    
    ufunc_dict = dict(input_core_dims=[['time']], output_core_dims=[['time']], vectorize=True,
                      output_dtypes=float, dask='parallelized')
    
    try:
        to_return = xr.apply_ufunc(func1d, da, **ufunc_dict)
    except ValueError as e:
        logger.debug(e)
        ufunc_dict.pop('output_dtypes')
        logger.debug(f'Trying again without output type specifiction {ufunc_dict}')
        to_return = xr.apply_ufunc(func1d, da, **ufunc_dict)
    
    return to_return
       

def trend_fit(da:xr.DataArray, method:str=None, order:int=1, lowess_window:int=30, func_kwargs:Optional[Dict]={},
             logginglevel='ERROR'):
    '''
    Generate a trend line for each grid cell in a given dataset.

    Parameters:
    da : xr.DataArray
        The data array to calculate the trend line along.
    method : str
        The method used to detrend. Options: [POLYNOMIAL, LOWESS]
    order : int
        Only for polynomial fitting. The order of the polynomial.
    lowess_window : int
        Only for LOWESS fitting. The window to take filter over.
    func_kwargs : dict, optional
        Additional keyword arguments for the detrending function.
    logginglevel : str
        The desired logging level.

    Returns:
    xr.DataArray
        The detrended data array.
    '''
    utils.change_logging_level(logginglevel)
    
    if not method:
        raise ValueError(f'method must be specified. Method options:  {[i.value for i in detrendingMethods]}')

    if da.chunks is not None:
        da = da.unify_chunks()
    
    logger.debug(f'{method=}\n data \n {da}')
    
    method = detrendingMethods[method.upper()]
    
    detrend_log_info = f'{order=}' if method == detrendingMethods.POLYNOMIAL else f'{lowess_window=}'
    logger.info(f'Detrending data using {method.name} with ' + detrend_log_info)
    
    if method == detrendingMethods.POLYNOMIAL:
        func1d = partial(polynomial_fit, order=order)

    elif method == detrendingMethods.LOWESS:
        func1d = partial(lowess, 
                         exog=np.arange(len(da.time.values)), frac=lowess_window/len(da.time.values), return_sorted=False)
    
    logger.debug(f'func1d = {func1d.func.__name__}\n{func1d}')
    da_trend = apply_detrend_as_ufunc(da, func1d, **func_kwargs)
    
    return da_trend



