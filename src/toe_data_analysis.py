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
from scipy.stats import spearmanr


from numpy.typing import ArrayLike
from dask.distributed import wait

# My imports
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts'))
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))

import toe_constants as toe_const
import utils
from utils import logger

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





def data_var_pattern_correlation_all_combs(ds: xr.Dataset, logginglevel='ERROR') -> pd.DataFrame:
    """
    Calculate Spearman correlation between variables in a dataset for all combinations.

    Parameters:
        ds (xr.Dataset): An xarray Dataset containing variables for correlation.

    Returns:
        pd.DataFrame: A DataFrame showing the Spearman correlation between all variables.
                      Rows and columns represent variables, values are correlation coefficients.
                      NaN is placed where the variables are the same or have already been correlated.
    """
    utils.change_logginglevel(logginglevel)
    
    data_vars = ds.data_vars
    tests_used: List[str] = list(data_vars)
    test_combinations = list(combinations(tests_used, 2))  # Generate unique combinations

    # Dictionary to store correlations
    correlations: Dict[str, Dict[str, float]] = {}
    for test_left, test_right in test_combinations:
        logger.info(f'{test_left} - {test_right}')

        ds_left = ds[test_left]
        ds_right = ds[test_right]

        # If one has member and the other does not e.g. sn_ens_med vs other
        # if 'member' in list(ds_left) and 'member' not in list(ds_left):
        if ('member' in ds_left) != ('member' in ds_right):
            if 'member' in ds_left:
                ds_left = ds_left.median(dim='member')
            else:
                ds_right = ds_right.median(dim='member')

        # Calculate Spearman correlation
        corr_values = spearmanr(
            ds_left.values.flatten(),
            ds_right.values.flatten(),
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
    
def xarray_data_var_pattern_correlation_all_combs(ds, logginlevel='ERROR'):
    return data_var_pattern_correlation_all_combs(ds, logginlevel).to_xarray()




    
def compute_region_metadata(only_1s_ds_region: xr.Dataset, max_points: xr.Dataset) -> dict:
    """Compute metadata for a region, including available and maximum points."""
    number_of_points = only_1s_ds_region.sum(dim=['lat', 'lon']).values.item()
    max_number_of_points = max_points.sum(dim=['lat', 'lon']).values.item()
    return {
        'available': number_of_points,
        'maximum': max_number_of_points
    }



def compute_weights(ds: xr.Dataset) -> xr.DataArray:
    """Compute weights based on latitude for area-weighted calculations."""
    weights = np.cos(np.deg2rad(ds.lat))
    weights.name = 'weights'
    return weights






def log_dataset(ds):
    """Compact single-line summary for xarray objects (for logging)."""
    if ds is None:
        return "None"
    try:
        dims_summary = ", ".join(f"{k}:{v}" for k, v in ds.dims.items())
        vars_summary = (
            ", ".join(f"{var}{tuple(ds[var].shape)}" for var in ds.data_vars)
            if isinstance(ds, xr.Dataset)
            else f"{ds.name}{tuple(ds.shape)}"
        )
        return f"dims=({dims_summary}) | vars=[{vars_summary}]"
    except Exception as e:
        return f"(unprintable: {e})"



def percent_emerged(toe_metric_ds, toe_ds, does_not_emerge_ds, time=None, return_all=False, logginglevel='INFO'):
    # Check for null input
    utils.change_logginglevel(logginglevel)
    logger.trace(log_dataset(toe_metric_ds))
    logger.trace(log_dataset(toe_ds))
    logger.trace(log_dataset(does_not_emerge_ds))

    if check_for_null(toe_metric_ds): return

    # print(toe_metric_ds, toe_ds, sep='\n---\n')

    # Ensure proper handling of xarray objects
    if not isinstance(toe_metric_ds, (xr.Dataset, xr.DataArray)):
        raise TypeError(f"Expected xarray.Dataset or xarray.DataArray, got {type(toe_metric_ds)}")
    
    # Ensure time dimension is valid
    if 'time' not in toe_metric_ds.dims:# or not hasattr(toe_metric_ds.time, 'dt'):
        raise ValueError("The 'time' dimension is missing or not a datetime array in toe_metric_ds.")
    time = toe_metric_ds.time.dt.year.values if time is None else time
    logger.debug(f'time=\n{time}')
    
    # Apply binary emergence calculation
    binary_emergence_ds = xr.apply_ufunc(
        calculate_returned_binary_ds,
        toe_metric_ds,
        toe_ds,
        time,
        input_core_dims=[['time'], [], ['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized'
    )
    binary_emergence_ds = binary_emergence_ds.compute()
    # Convert values to binary (1 or 0)
    binary_emergence_ds = xr.where(binary_emergence_ds == 1, 1, 0)

    # Only return where the dataset does emerge (seems to be issue for some reason)
    binary_emergence_ds = binary_emergence_ds.where(does_not_emerge_ds == 0)
    logger.trace(log_dataset(binary_emergence_ds))
    
    # Compute weights and percentage emergence
    only_1s_ds = xr.ones_like(binary_emergence_ds.isel(time=0))
    logger.trace(log_dataset(only_1s_ds))

    weights = compute_weights(binary_emergence_ds)
    logger.trace(log_dataset(weights))
    
    emergence_time_series_ds = percentage_lat_lons(
        binary_emergence_ds, only_1s_ds, weights
    )
    logger.trace(log_dataset(emergence_time_series_ds))

    if return_all: return emergence_time_series_ds, binary_emergence_ds, only_1s_ds
    
    return emergence_time_series_ds







def check_for_null(ds):  return not ds.dims and not ds.data_vars


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



def calculate_returned_binary_ds(year_of_emergence, time_years):
    """
    Calculates a binary array representing when emergence has occurred.

    Parameters
    ----------
    year_of_emergence : float or int
        The year of emergence at a grid cell.
    time_years : array-like
        Array of time coordinates (years) to compare against.

    Returns
    -------
    np.ndarray
        Binary array (same length as time_years), with 0 before emergence,
        and 1 from the year of emergence onward.
    """

    # --- Fix for xarray apply_ufunc wrapping kwargs in tuples ---
    if isinstance(time_years, tuple):
        time_years = time_years[0]
    # time_years = np.asarray(time_years)

    # Handle missing or invalid input
    if np.isnan(year_of_emergence): return np.full_like(time_years, np.nan)

    # Ensure integer comparison
    year_of_emergence = int(year_of_emergence)
    time_years = time_years.astype(int)

    # Find where emergence occurs in the time axis
    if year_of_emergence not in time_years:
        raise RuntimeError(
            f"Year of emergence {year_of_emergence} not in available years {time_years}"
        )

    idx = np.where(time_years == year_of_emergence)[0][0]

    # Create binary array: 0 before, 1 after emergence
    binary = np.zeros_like(time_years, dtype=int)
    binary[idx:] = 1
    return binary



from numba import njit

@njit
def calculate_returned_binary_ds_jit(year_of_emergence, time_years):
    if np.isnan(year_of_emergence):
        return np.full(time_years.shape, np.nan)
    yoe = int(year_of_emergence)
    idx = -1
    for i in range(time_years.size):
        if int(time_years[i]) == yoe:
            idx = i
            break
    if idx == -1:
        raise RuntimeError("Year of emergence not in available years")
    binary = np.zeros(time_years.shape, dtype=np.int32)
    binary[idx:] = 1
    return binary



def percent_emerged_regions(
    toe_ds: xr.Dataset,
    does_not_emerge_ds: xr.Dataset,
    land_mask_ds: xr.DataArray,
    years: np.ndarray,
    regions: list = None,
    return_all: bool = False,
    logginglevel: str = "ERROR",
    debug: bool = False,
):
    """
    Compute regional time series of percent area that has emerged, using
    binary time-series construction for each grid cell.

    Each grid cell's time series is 0 before its year of emergence and 1
    thereafter. For each region, a weighted spatial mean of these binary
    fields is taken at each time step to yield the fraction of area emerged.

    Parameters
    ----------
    toe_ds : xr.Dataset
        Dataset containing the year of emergence for each metric (lat–lon grids).
        Values are years of emergence; NaN where no emergence occurs.
    does_not_emerge_ds : xr.Dataset
        Binary mask indicating where emergence does *not* occur
        (0 = emergence occurs, 1 = no emergence).
    land_mask_ds : xr.DataArray
        Boolean land–sea mask (True = land, False = ocean) used for subsetting.
    years : array-like
        1D list or array of years to evaluate (e.g., 1850–2100).
    regions : list, optional
        List of regional definitions (defaults to `toe_const.regionLatLonTuples`).
    return_all : bool, optional
        If True, return a list of intermediate regional datasets before concatenation.
    logginglevel : str, optional
        Logging level ('ERROR', 'INFO', 'DEBUG', 'TRACE').
    debug : bool, optional
        If True, also return intermediate region results (for inspection).

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ('region', 'time') and one variable per metric,
        representing the percentage of area emerged at each time step.
    """
    def __prepare_region_datasets(
        binary_emergence_ds: xr.Dataset,
        only_1s_ds: xr.Dataset,
        land_mask_ds: xr.Dataset,
        region_name: str,
        lat_slice: slice)-> tuple:
        
        """
        Prepare datasets for a specific region, applying masking if needed.
        """
        
        ds_region = binary_emergence_ds.sel(lat=lat_slice).expand_dims({'region': [region_name]})
        only_1s_ds_region = only_1s_ds.sel(lat=lat_slice)
    
        if region_name in ['land', 'ocean']:
            logger.debug(f'Applying mask for {region_name=}')
            mask_to_use_ds = xr.where(land_mask_ds, 0, 1) if region_name == 'land' else xr.where(land_mask_ds, 1, 0)
            logger.debug(f'\n{ds_region}\n')
    
            ds_region = ds_region.where(mask_to_use_ds)
            only_1s_ds_region = only_1s_ds_region.where(mask_to_use_ds)
    
            logger.debug(f'\n{ds_region}\n')
        return ds_region, only_1s_ds_region
        
    
    utils.change_logginglevel(logginglevel)
    logger.info("Starting percent_emerged_regions computation")

    if check_for_null(toe_ds):
        logger.warning("toe_ds is empty — returning None.")
        return

    # --- Binary emergence time series ---
    logger.info("Constructing binary emergence dataset via apply_ufunc")

    binary_emergence_ds = xr.apply_ufunc(
        calculate_returned_binary_ds,
        toe_ds,
        input_core_dims=[[]],
        output_core_dims=[["time"]],
        kwargs={"time_years": years},
        output_sizes={'time': len(years)},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[int]
    )


    binary_emergence_ds = binary_emergence_ds.assign_coords(time=years)
    # binary_emergence_ds = binary_emergence_xds.compute()

    logger.debug(f"Binary dataset computed: dims={binary_emergence_ds.dims}")

    # Convert values to binary (1/0) and apply 'does not emerge' mask
    binary_emergence_ds = xr.where(binary_emergence_ds == 1, 1, 0)
    binary_emergence_ds = binary_emergence_ds.where(does_not_emerge_ds == 0)

    # Compute spatial weights
    weights = compute_weights(toe_ds)
    logger.debug(f"Computed weights: shape={weights.shape}, sum={weights.sum().item():.3e}")

    # Template of ones (used for normalisation in percentage_lat_lons)
    only_1s_ds = xr.ones_like(binary_emergence_ds.isel(time=0))

    if regions is None:
        regions = toe_const.regionLatLonTuples
        logger.info(f"Using default region set ({len(regions)} total regions)")


    futures = [binary_emergence_ds, only_1s_ds, weights]
    wait(futures)

    
    ds_collection = []

    # --- Region loop ---
    for region in regions:
        region_name = region.name.lower()
        logger.info(f"--- Processing region: {region.name} ---")
        logger.debug(f"Lat–lon bounds: {region.value.latlon}")

        # Subset the lat lon, and apply mask if necessary
        ds_region, only_1s_ds_region = __prepare_region_datasets(
            binary_emergence_ds, only_1s_ds, land_mask_ds, region_name,region.value.latlon)

        if not ds_region.data_vars:
            logger.warning(f"No valid data after subsetting {region.name}, skipping.")
            continue

        logger.trace(f"Subset region dataset shape: {ds_region[list(ds_region.data_vars)[0]].shape}")

        # Compute percentage emergence over time for this region
        try:
            time_series_ds = percentage_lat_lons(ds_region, only_1s_ds_region, weights)
        except Exception as e:
            logger.error(f"Error computing percentage for region {region.name}: {e}")
            continue

        if check_for_null(time_series_ds):
            logger.warning(f"Result for {region.name} is empty, skipping.")
            continue

        ds_collection.append(time_series_ds)
        logger.debug(f"Completed region {region.name}")

    if debug:
        return ds_collection

    if not ds_collection:
        logger.warning("No regional results were produced — check input data or masks.")
        return

    # --- Combine regional results ---
    emergence_time_series_ds = xr.concat(ds_collection, dim="region")
    logger.info(f"Combined {len(ds_collection)} regions into final dataset")
    logger.debug(f"Final dataset dimensions: {emergence_time_series_ds.dims}")
    logger.trace(f"Region names: {emergence_time_series_ds.region.values}")

    logger.info("percent_emerged_regions complete")
    return emergence_time_series_ds




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
    if np.isnan(year_of_emergence): return np.nan

    # The is useful when doing enesemble median, this can result in values that have decimals
    if isinstance(year_of_emergence, float): year_of_emergence = int(year_of_emergence)
    
    # Find the index of year_of_emergence in time_years
    emergence_arg = np.argwhere(time_years == year_of_emergence)
    
    emergence_arg = emergence_arg.item()
    
    # Get the value in arr at the emergence_arg index
    value_at_arg = arr[emergence_arg]
    
    return value_at_arg



def compute_pct_emergence(toe_values, percentile=50):
    """
    Determines the interpolated year when 50% of ensemble members have emerged.
    
    Parameters:
    toe_values (list or np.array): Array of emergence years, with NaN for non-emerging members.
    
    Returns:
    float: Interpolated year when 50% of members have emerged, or NaN if fewer than 50% emerge.
    """
    toe_values = np.array(toe_values)
    N_threshold = percentile/100 * len(toe_values)  # 50% threshold
    valid_toe = np.sort(toe_values[~np.isnan(toe_values)])
    
    if len(valid_toe) < np.ceil(N_threshold): return np.nan
    
    idx = N_threshold - 1  # 0-based index

    # This occurs when t
    if idx.is_integer():
        return valid_toe[int(idx)]
    
    lower = valid_toe[int(idx)]
    upper = valid_toe[int(idx) + 1]
    weight = idx - int(idx)
    interpolated_value = lower + weight * (upper - lower)
    
    return interpolated_value


# def percent_emerged_regins(binary_emergence_ds: xr.Dataset, land_mask_ds: xr.Dataset, 
#                                          only_1s_ds: xr.Dataset, regions: list, logginglevel='ERROR') -> xr.Dataset:
#     """
#     Calculate the percentage of stability for regions based on binary emergence data.

#     Parameters:
#         binary_emergence_ds (xr.Dataset): Dataset containing binary emergence data.
#         land_mask_ds (xr.Dataset): Dataset containing land mask data.
#         only_1s_ds (xr.Dataset): Dataset with only values of 1 for valid data points.
#         regions (list): List of region objects containing a name and latitude slice.
#         logginglevel (str): Logging level for the function. Default is 'ERROR'.

#     Returns:
#         xr.Dataset: Dataset with percentage of stability for each region.
#     """
#     utils.change_logginglevel(logginglevel)

#     weights = compute_weights(binary_emergence_ds)
#     ds_collection = []
#     points_in_region_dict = {}

#     for region in regions:
#         region_name, lat_slice = region['name'].lower(), region['slice']

#         ds_region, only_1s_ds_region, max_points = prepare_region_datasets(
#             binary_emergence_ds, only_1s_ds, land_mask_ds, region_name, lat_slice
#         )

#         time_series_ds = percentage_lat_lons(ds_region, only_1s_ds_region, weights)
#         ds_collection.append(time_series_ds)

#         points_in_region_dict[region_name] = compute_region_metadata(
#             only_1s_ds_region, max_points
#         )

#     emergence_time_series_ds = xr.concat(ds_collection, dim='region')
#     emergence_time_series_ds.attrs = points_in_region_dict
    
#     return emergence_time_series_ds







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


