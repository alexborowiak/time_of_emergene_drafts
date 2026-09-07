import numpy as np
import xarray as xr
from dask.distributed import wait

import os, sys
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts'))
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))
# import toe_constants as toe_const
import toe_calc
import my_stats



def _construct_rolling_window_dim(
    data_ds: xr.DataArray,
    window: int = 30
) -> xr.DataArray:
    """
    Compute rolling window along the time dimension and construct 'window_dim'.
    """
    data_ds_window = (
        data_ds
        .rolling(time=window, center=True, min_periods=window)
        .construct('window_dim')
        .persist()
    )
    wait(data_ds_window)
    return data_ds_window


def _create_base_period_kde(
    base_period_ds: xr.DataArray,
    x: np.ndarray,
    kde_kwargs: dict | None = None
) -> xr.DataArray:
    """
    Compute the baseline KDE along the 'time' dimension.
    """
    kde_kwargs = dict(bw_method=0.2) if kde_kwargs is None else kde_kwargs

    if base_period_ds.chunks:
        dask_kwargs = dict(dask='parallelized', output_dtypes=float)
    else: dask_kwargs = dict()
        
    
    base_period_kde = xr.apply_ufunc(
        toe_calc.create_kde_x_exists,
        base_period_ds,
        input_core_dims=[['time']],
        output_core_dims=[['x']],
        kwargs={'x': x, **kde_kwargs},
        vectorize=True,
        dask_gufunc_kwargs={'output_sizes': {'x': len(x)}},
        keep_attrs=True,
        **dask_kwargs
    ).persist()
    wait(base_period_kde)
    return base_period_kde

def __overlap_apply(
    func,
    data_ds: xr.DataArray | None = None,
    base_period_ds: xr.DataArray = None,
    data_ds_window: xr.DataArray | None = None,
    base_period_kde: xr.DataArray | None = None,
    kde_kwargs: dict | None = None,
    window: int = 30,
    num_points: int = 1000,
    x: np.ndarray | None = None
) -> xr.DataArray:
    """
    Apply a distribution comparison function (e.g., FGA or Hellinger distance)
    between a rolling window and a baseline KDE.

    Parameters
    ----------
    func : callable
        Function that compares two distributions (data window and KDE).
    data_ds : xr.DataArray | None
        Original data; used to compute rolling window if data_ds_window not supplied.
    base_period_ds : xr.DataArray
        Baseline data for KDE computation.
    data_ds_window : xr.DataArray | None
        Precomputed rolling window; if None, it is constructed.
    base_period_kde : xr.DataArray | None
        Precomputed baseline KDE; if None, it is created.
    kde_kwargs : dict | None
        Options for KDE computation (e.g., {'bw_method': 0.2}).
    window : int
        Rolling window size.
    num_points : int
        Number of points in the x-grid.
    x : np.ndarray | None
        Optional x-grid to use for KDEs; if None, it is computed from data.

    Returns
    -------
    xr.DataArray
        Resulting metric (FGA or Hellinger distance) along the rolling window.
    """

    kde_kwargs = dict(bw_method=0.2) if kde_kwargs is None else kde_kwargs

    # Compute rolling window if not supplied
    if data_ds_window is None:
        data_ds_window = _construct_rolling_window_dim(data_ds, window)

    # Compute x-grid if not supplied
    if x is None:
        data_max = data_ds_window.max().persist().values.item()
        data_min = data_ds_window.min().persist().values.item()
        x = toe_calc.create_x(bmin=data_min, bmax=data_max, num_points=num_points)

    # Compute baseline KDE if not supplied
    if base_period_kde is None:
        base_period_kde = _create_base_period_kde(base_period_ds, x, kde_kwargs)

    if data_ds_window.chunks:
        dask_kwargs = dict(dask='parallelized', output_dtypes=float)
    else: dask_kwargs = dict()

    # Apply the function
    geom_ds = xr.apply_ufunc(
        func,
        data_ds_window,
        base_period_kde,
        input_core_dims=[['window_dim'], ['x']],
        exclude_dims={'window_dim'},
        kwargs={'x': x, 'method_kwargs': kde_kwargs},
        vectorize=True,
        keep_attrs=True,
        **dask_kwargs
    ).compute()

    # Minimal metadata
    geom_ds.attrs['bw_method'] = kde_kwargs.get('bw_method', 0.2)

    return geom_ds


def fga(*args, **kwargs):
    frac_geom_ds = __overlap_apply(toe_calc.fractional_geometric_area_optimized, *args, **kwargs)
    frac_geom_ds.name = 'frac'
    return frac_geom_ds

# inherit docstring
fga.__doc__ = __overlap_apply.__doc__


def hd(*args, **kwargs):
    geom_ds = __overlap_apply(toe_calc.hellinger_distance_optimized, *args, **kwargs)
    geom_ds.name = 'hd'
    return geom_ds

# inherit docstring
hd.__doc__ = __overlap_apply.__doc__
    

def __statistical_hyp_apply(
    data_ds_window: xr.DataArray,
    base_period_window_ds: xr.DataArray,
    func: callable
) -> xr.DataArray:
    """
    Apply a statistical test function to two rolling window DataArrays
    along 'window_dim'.
    """

    rolling_window_kwargs = dict(
        input_core_dims=[['window_dim'], ['window_dim']],
        exclude_dims={'window_dim'},
        vectorize=True,
        dask='parallelized'
    )

    out_ds = xr.apply_ufunc(
        func,
        data_ds_window,
        base_period_window_ds,
        **rolling_window_kwargs
    ).compute()

    return out_ds


def ttest(
    data_ds_window: xr.DataArray,
    base_period_window_ds: xr.DataArray
) -> xr.DataArray:
    """
    Compute t-test p-values between data and baseline windows.
    """

    out_ds = __statistical_hyp_apply(
        data_ds_window,
        base_period_window_ds,
        func=toe_calc.return_ttest_pvalue
    )

    out_ds.name = 'ttest'

    return out_ds


def ks(
    data_ds_window: xr.DataArray,
    base_period_window_ds: xr.DataArray
) -> xr.DataArray:
    """
    Compute KS-test p-values between data and baseline windows.
    """

    ks_ds = __statistical_hyp_apply(
        data_ds_window,
        base_period_window_ds,
        func=toe_calc.return_ks_pvalue
    )

    ks_ds.name = 'ks'

    return ks_ds


def sn_ratio(data_ds, start=0, end=30, window=30):
    
    base_period_ds = data_ds.isel(time=slice(start, end))
    
    data_anom_ds = data_ds - base_period_ds.mean(dim='time')
    base_period_anom_ds = base_period_ds - base_period_ds.mean(dim='time')
    
    ds_signal_lowess = xr.apply_ufunc(
        my_stats.apply_lowess, 
        data_anom_ds,#.chunk({'time':-1, 'lat':10}), 
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True, 
        kwargs = dict(window=41),
        dask='parallelized',
        output_dtypes=[float]
    ).compute()

    ds_signal_lowess.name = 'signal'
    
    # Noise series is detrended data
    ds_noise_series_lowess = (data_anom_ds - ds_signal_lowess).compute()
    ds_noise_series_lowess.name = 'noise'
    
    ds_noise_lowess_base_period = ds_noise_series_lowess.isel(time=slice(start, end)).std(dim='time')

    ds_noise_full = ds_noise_series_lowess.std(dim='time')

    ds_std_roll = ds_noise_series_lowess.rolling(time=window, center=True).std(dim='time')

    ds_noise_roll = np.sqrt(1/2*(ds_std_roll**2+ds_noise_lowess_base_period**2))
        
    ds_sn_lowess_base_period = ds_signal_lowess/ds_noise_lowess_base_period
    ds_sn_lowess_full = ds_signal_lowess/ds_noise_full
    ds_sn_lowess_roll = ds_signal_lowess/ds_noise_roll

    
    ds_sn_lowess_base_period.name = 'sn_lowess_base'
    ds_sn_lowess_full.name = 'sn_lowess_full'
    ds_sn_lowess_roll.name = 'sn_roll'
    ds_noise_roll.name = 'noise_roll'
    
    out_ds =  xr.merge(
        [ds_sn_lowess_base_period , ds_sn_lowess_full, ds_sn_lowess_roll,
         ds_noise_series_lowess, ds_signal_lowess])

    return out_ds
# def fga(data_ds, base_period_ds, data_ds_window=None, base_period_kde=None, kde_kwargs=None, window:int=30):

#     kde_kwargs= dict(bw_method=0.2) if kde_kwargs is None else kde_kwargs # silverman, scott#bw_method=0.2)

#     # The x-values for the KDE are based upon the max and min
#     data_max = data_ds.max().persist().values.item()
#     data_min = data_ds.min().persist().values.item()
#     num_points = 1000
#     x = toe_calc.create_x(bmin=data_min, bmax=data_max, num_points=num_points) 

#     if data_ds_window is None:
#         data_ds_window = (data_ds
#                   .rolling(time=window, center=True, min_periods=window)
#                   .construct('window_dim')
#                   .persist()
#                  ) 
#         wait(data_ds_window);

#     if base_period_kdebase_period_kde is None:
#         base_period_kde = xr.apply_ufunc(
#             toe_calc.create_kde_x_exists,
#             base_period_ds,
#             input_core_dims=[['time'], ],
#             output_core_dims=[['x']],
#             kwargs={'x': x, **kde_kwargs},
#             vectorize=True,
#             dask='parallelized',
#             dask_gufunc_kwargs={'output_sizes': {'x': len(x)}},
#             # output_sizes={'x':len(x)},  # Specify the size of the 'bin' dimension
#             output_dtypes=float
#         ).persist()
    
#         wait(base_period_kde);
    
#     frac_geom_ds = xr.apply_ufunc(
#         toe_calc.fractional_geometric_area_optimized,
#         data_ds_window,
#         base_period_kde,
#         input_core_dims=[['window_dim'], ['x']],
#         exclude_dims={'window_dim'},
#         kwargs={'x': x, 'method_kwargs':kde_kwargs},
#         vectorize=True,
#         dask='parallelized',
#         output_dtypes=float
#     ).compute()
#     # wait(frac_geom_ds2)
    
#     x_attrs = { 'bmin': data_min, 'bmax':data_max, 'num_points': num_points}
#     frac_geom_ds.attrs = {**frac_geom_ds.attrs, **kde_kwargs, **x_attrs}

#     frac_geom_ds.name = 'frac'
    
#     return frac_geom_ds
