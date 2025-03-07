import numpy as np
import xarray as xr
from dask.distributed import wait

import os, sys
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts'))
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))
# import toe_constants as toe_const
import toe_calc
import my_stats

def fga(data_ds, base_period_ds, data_ds_window):
    # The x-values for the KDE are based upon the max and min
    data_max = data_ds.max().persist().values.item()
    data_min = data_ds.min().persist().values.item()
    num_points = 1000
    x = toe_calc.create_x(bmin=data_min, bmax=data_max, num_points=num_points)
    
    
    kde_kwargs= dict(bw_method=0.2) # silverman, scott#bw_method=0.2)
    
    base_period_kde = xr.apply_ufunc(
        toe_calc.create_kde_x_exists,
        base_period_ds,
        input_core_dims=[['time'], ],
        output_core_dims=[['x']],
        kwargs={'x': x, **kde_kwargs},
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={'output_sizes': {'x': len(x)}},
        # output_sizes={'x':len(x)},  # Specify the size of the 'bin' dimension
        output_dtypes=float
    ).persist()
    wait(base_period_kde);
    
    frac_geom_ds = xr.apply_ufunc(
        toe_calc.fractional_geometric_area_optimized,
        data_ds_window,
        base_period_kde,
        input_core_dims=[['window_dim'], ['x']],
        exclude_dims={'window_dim'},
        kwargs={'x': x, 'method_kwargs':kde_kwargs},
        vectorize=True,
        dask='parallelized',
        output_dtypes=float
    ).compute()
    # wait(frac_geom_ds2)
    
    x_attrs = { 'bmin': data_min, 'bmax':data_max, 'num_points': num_points}
    frac_geom_ds.attrs = {**frac_geom_ds.attrs, **kde_kwargs, **x_attrs}

    frac_geom_ds.name = 'frac'
    
    return frac_geom_ds



def ks(data_ds_window, base_period_window_ds):
    # The arguements needed for all of the calculations
    rolling_window_kwargs = dict(
        input_core_dims=[['window_dim'], ['window_dim']],
        exclude_dims={'window_dim'},
        vectorize=True,
        dask='parallelized')
    
    ks_ds = xr.apply_ufunc(
            toe_calc.return_ks_pvalue,
            data_ds_window,
            base_period_window_ds,
        **rolling_window_kwargs
        ).compute()

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

    
    ds_sn_lowess_base_period.name = 'sn'
    ds_sn_lowess_full.name = 'sn_lowess_full'
    ds_sn_lowess_roll.name = 'sn_roll'
    ds_std_roll.name = 'noise_roll'
    
    out_ds =  xr.merge(
        [ds_sn_lowess_base_period , ds_sn_lowess_full, ds_sn_lowess_roll,
         ds_noise_series_lowess, ds_std_roll, ds_signal_lowess])

    return out_ds
