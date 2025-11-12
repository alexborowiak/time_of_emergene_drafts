

import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
from typing import List, Tuple, Optional
from matplotlib import gridspec
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import os, sys

# My imports
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts'))
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))

from toe_plots import NAME_MAPPING
import toe_constants as toe_const
import toe_plots
import utils
import plotting_utils

from utils import logger
##### Maps


from matplotlib.colors import ListedColormap, BoundaryNorm

cmap_binary = ListedColormap([
    (0, 0, 0, 0),       # transparent
    (0.8, 0.8, 0.8, 1)  # light grey
])

# hard bins: 0 maps to first color, 1 maps to second
norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap_binary.N)

no_emergence_plot_kwargs = dict(
    cmap=cmap_binary,
    norm=norm,
    add_colorbar=False
)


def plot_time_of_emergence_maps_main(
    toe_tree,
    emergence_time_series_50pct_tree,
    does_not_emerge_tree,
    data_availability_tree,
    PLOT_LEVELS,
    # Default parameters
    tests_subset_used=None,
    tests_var_used=None,
    fontscale=1.1,
    cmap=plt.cm.RdYlBu,
    cmap_binary=None,
    toe_emergence_levels=None,

):
    """
    Plots time of emergence (ToE) maps and bottom-panel time series
    for a set of variables and metrics.

    Parameters
    ----------
    toe_tree : xarray Dataset or dict-like
        Your tree holding the time-of-emergence (ToE) data.
    emergence_time_series_50pct_tree : xarray Dataset or dict-like
        Tree holding global emergence time series data.
    does_not_emerge_tree : xarray Dataset or dict-like
        Holds data indicating where emergence does not occur.
    data_availability_tree : xarray Dataset or dict-like
        Indicates data availability.
    tests_subset_used : list, optional
        The metrics to be shown along the rows. Defaults to ['sn_lowess_base', 'ks', 'frac'].
    tests_var_used : list, optional
        The variables to be shown along the columns. Defaults to list of keys in toe_tree.
    fontscale : float, optional
        Scaling factor for fonts.
    cmap : matplotlib Colormap, optional
        Colormap to use for the ToE map. Defaults to plt.cm.RdYlBu.
    Returns
    -------
    fig : matplotlib Figure
        The generated figure.
    axes : 2D array of matplotlib Axes
        Axes for the map panels (shape = [len(tests_subset_used), len(tests_var_used)]).
    bottom_axes : list of matplotlib Axes
        Axes for the bottom time series panels.
    """
    # Default fallbacks if None

    if tests_var_used is None:
        tests_var_used = list(toe_tree.keys())
    if tests_subset_used is None:
        tests_subset_used = list(toe_tree[tests_var_used[0]])
    if cmap_binary is None:
        # Provide a default fallback or raise an error if necessary
        cmap_binary = plt.cm.binary

    fig = plt.figure(figsize=(6*len(list(tests_var_used)), 3.7*len(tests_subset_used)))

    gs = gridspec.GridSpec(3, 3, height_ratios=[1*len(tests_subset_used)]+[0.2]+[1], hspace=0.1)
    
    grid_gs = gridspec.GridSpecFromSubplotSpec(
        len(tests_subset_used), len(tests_var_used), subplot_spec=gs[0, :], hspace=0)
    
    axes = np.array([
        [fig.add_subplot(grid_gs[row, col], projection=ccrs.PlateCarree()) for col in np.arange(len(tests_var_used))]
        for row in range(len(tests_subset_used))
    ])
    
    bottom_axes = [fig.add_subplot(gs[-1, col]) for col in np.arange(len(tests_var_used))]#

    for col, variable in enumerate(tests_var_used):
        print(f"\nPlotting {variable}: ", end="")
        
        toe_plot_kwargs = dict(
            add_colorbar=False, levels=PLOT_LEVELS.get(variable, np.arange(1920, 2030, 10))
        )

        for row, metric in enumerate(tests_subset_used):
            print(f"{metric} ", end="")

            ax = axes[row, col]
            if col == 0:
                row_label = toe_plots.METRIC_MAP.get(metric, metric)
                if 'Ratio' in row_label: row_label = row_label.replace(' Ratio', '\nRatio')
                axes[row, 0].annotate(
                    row_label, xy=(-0.27, 0.5),
                    xycoords='axes fraction', rotation=0, ha='center', va='center',
                    fontsize=12*fontscale)
            else:
                ax.set_title('')

            # Varialbe is not available for dataset (e.g. piControl for best)
            if metric not in list(toe_tree[variable].to_dataset()): 
                ax.axis('off')
                continue
            
            toe_ds = toe_tree[variable].to_dataset()[metric]

         
            emergence_time_series_50pct_ds = (
                emergence_time_series_50pct_tree[variable]
                .to_dataset()
                .sel(region='global')
            )

            if 'member' in toe_ds.coords:
                toe_ds = toe_calc.calculate_percent_member_emergence(toe_ds).compute() #.median(dim='member')
                does_not_emerge_da = xr.where(np.isnan(toe_ds), 1, 0)

            else:
  
                does_not_emerge_da = does_not_emerge_tree[variable].to_dataset()[metric]
                if 'best' in variable:
                    plotting_utils.hatch(
                        ax,
                        data_availability_tree[variable].to_dataset().to_array().squeeze(),
                        **plotting_utils.no_data_plot_kwargs
                    )

            max_value  = np.max(toe_plot_kwargs['levels']-15)
            toe_ds = xr.where(toe_ds >= max_value, max_value, toe_ds)

            does_not_emerge_da.plot(ax=ax, **no_emergence_plot_kwargs)
            toe_ds.plot(cmap=cmap, ax=ax, **toe_plot_kwargs)
            
            ax.coastlines()
            plotting_utils.add_lat_markers(ax, side='right', fontscale=fontscale)

     
            if col > 0: ax.set_title('')

  
            if row == 0:
                ax.set_title(
                    toe_plots.NAME_MAPPING.get(variable, variable),
                    fontsize=14*fontscale
                )

        # Now add the time series in the bottom panel
        bax = bottom_axes[col]
        dataset_for_series = (emergence_time_series_50pct_ds.median(dim='member')
                              if 'member' in list(emergence_time_series_50pct_ds.coords)
                              else emergence_time_series_50pct_ds)
        
        toe_plots.percent_emerged_series(dataset_for_series,
                                         tests_subset_used, ax=bax, legend=False,
                                         fontscale=fontscale)

        bax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        bax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
        bax.tick_params(axis='x', rotation=45)
        bax.tick_params(axis='x', which='major', length=6, width=1)
        bax.tick_params(axis='x', which='minor', length=3, width=1)
        bax.set_xlim(np.take(toe_plot_kwargs['levels'], [0, -1]))
        
    for bax in bottom_axes: bax.set_xlabel("Time", fontsize=14*fontscale)
    bottom_axes[0].set_ylabel("Percent of Surface\nArea Emerged", fontsize=14*fontscale)
    leg = bottom_axes[0].legend(ncol=4, loc='center', bbox_to_anchor=(1.5, -.65),
                                fontsize=10*fontscale)

    cbar1 = plotting_utils.create_discrete_colorbar(
        cmap, PLOT_LEVELS['best_tas'], plt.subplot(gs[-2, :]) ,
        'Time of Emergence', fontscale=fontscale, orientation='horizontal', pad=7
    )

    return fig, axes, bottom_axes, leg


def plot_time_of_emergence_maps_single_var_with_iqr(
    toe_ds,
    does_not_emerge_ds,
    data_availability_ds,
    tests_subset_used=None,
    fig=None, axes=None, caxes=None, ncols = 2,
    fontscale=1.1,
    levels:np.ndarray = np.arange(1920, 2030, 10),
    iqr_levels:np.ndarray = np.arange(0, 16, 1),
    cmap=plt.cm.terrain_r,
):
    """
    Plots time of emergence (ToE) maps and bottom-panel time series
    for a single variable and multiple metrics.
    """

    if tests_subset_used is None: tests_subset_used = list(toe_ds[tests_var_used])

    nrows = len(tests_subset_used)
    if fig is None:
        fig = plt.figure(figsize=(3.7*ncols, 2 * nrows))
    if axes is None:
        gs = gridspec.GridSpec(2, 1, hspace=0.1, height_ratios=[1, 0.05], wspace=0.2)
        axes_gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[0])
        axes = np.array([
            [fig.add_subplot(axes_gs[row, col], projection=ccrs.PlateCarree())
             for col in range(ncols)] for row in range(nrows)])
    if caxes is None:
        caxes_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[-1])
        caxes = [plt.subplot(caxes_gs[i]) for i in [-2, -1]]
    

    toe_plot_kwargs = dict(add_colorbar=False, levels=levels)

    toe_ds_iqr = (toe_ds.quantile(0.75, dim='member') - toe_ds.quantile(0.25, dim='member')).compute()

    for row, metric in enumerate(tests_subset_used):
        print(f"{metric} ", end="")

        if 'member' in list(toe_ds[metric].coords):
            toe_ds_median = toe_calc.calculate_percent_member_emergence(toe_ds[metric]).compute()
            does_not_emerge_da = xr.where(np.isnan(toe_ds_median), 1, 0)
            does_not_emerge_da.plot(ax=axes[row, 0], **no_emergence_plot_kwargs)
            toe_ds_median.plot(cmap=cmap, ax=axes[row, 0], **toe_plot_kwargs, extend='max')
            
            toe_ds_iqr[metric].plot(cmap=cmap, ax=axes[row, 1], levels=iqr_levels, add_colorbar=False, extend='max')
            local_axes = [axes[row, 0], axes[row, 1]]

        
        else:
            toe_ds[metric].plot(cmap=cmap, ax=axes[row, 0], **toe_plot_kwargs, extend='max')
            local_axes = [axes[row, 0]]

        for ax in local_axes:
            ax.coastlines()
            plotting_utils.add_lat_markers(ax, side='right')
   
        axes[row, 0].annotate(
            toe_plots.METRIC_MAP.get(metric, metric).replace('Ratio', '\nRatio'), xy=(-0.29, 0.5),
            xycoords='axes fraction', rotation=0, ha='center', va='center',
            fontsize=10*fontscale)

        [axes[row, col].set_title('') for col in range(ncols)]


    cbar_kwargs = dict(fontscale=fontscale, orientation='horizontal', pad=7, extend='max')
    
    cbar1 = plotting_utils.create_discrete_colorbar(
        cmap, levels, caxes[0],'Time of Emergence', tick_slice=slice(None, None, 4) ,  **cbar_kwargs)
    cbar2 = plotting_utils.create_discrete_colorbar(
        cmap, iqr_levels, caxes[1],'Interquartile Range', tick_slice=slice(None, None, 4), **cbar_kwargs)

    axes[0, 0].set_title('Median')
    axes[0, 1].set_title('Inter-Quartile Range')
    
    # for ax in axes.ravel():
        # ax.coastlines()
        # plotting_utils.add_lat_markers(ax, side='right')
        
    return fig, axes, bottom_axes

def plot_time_of_emergence_maps_single_var(
    toe_ds,
    does_not_emerge_ds=None,
    data_availability_ds=None,
    tests_subset_used=None,
    fontscale=1.1,
    levels:np.ndarray = np.arange(1920, 2030, 10),
    cmap=plt.cm.RdYlBu,
    toe_emergence_levels=None,
    fig = None,
    gs= None,
    axes = None,
    cax  = None,
    ncol=3,
    cbar_title='Time of Emergence',

):
    """
    Plots time of emergence (ToE) maps and bottom-panel time series
    for a single variable and multiple metrics.
    """

    if tests_subset_used is None:
        tests_subset_used = list(toe_ds[tests_var_used])

    num_rows = int(np.ceil(len(tests_subset_used)/ ncol))


    if fig is None:
        fig = plt.figure(figsize=(5 * ncol, 3.2 * num_rows))
        
        # Create GridSpec
        gs = gridspec.GridSpec(
            num_rows + 1, ncol,  # +1 for the colorbar row
            height_ratios=[1] * num_rows + [0.1],  # Last row reserved for colorbar
            wspace=0.2, hspace=0.2
        )

    if axes is None:

        # Compute empty slots
        total_slots = num_rows * ncol
        diff = total_slots - len(tests_subset_used)
        
        print(f"Empty slots: {diff}, Total plots: {len(tests_subset_used)}")
        
        # Compute positions to fill
        positions_to_fill = list(range(len(tests_subset_used)))
          # Fill normally
        
        if diff == 1:
            # Fill leftmost and rightmost on last row
            positions_to_fill = positions_to_fill[:-1]
            leftmost = total_slots - ncol
            rightmost = total_slots - 1
            positions_to_fill += [leftmost+1, rightmost+1]
        elif diff == 2:
            positions_to_fill = positions_to_fill[:-1]
            # Place single plot in the center of last row
            mid_position = total_slots - ncol + (ncol // 2)
            positions_to_fill.append(mid_position)
  
        print("Final positions to fill:", positions_to_fill)
    
        # Create Axes with the specified positions
        axes = np.array([fig.add_subplot(gs[num], projection=ccrs.PlateCarree()) for num in positions_to_fill])
        
        # Create Colorbar Axis
        cax = plt.subplot(gs[-1, :])  # Colorbar at the bottom

    toe_plot_kwargs = dict(
        add_colorbar=False, levels=levels,
    )

    for num, metric in enumerate(tests_subset_used):
        print(f"{metric} ", end="")

        ax = axes[num]

        if does_not_emerge_ds: does_not_emerge_ds[metric].plot(ax=ax, **no_emergence_plot_kwargs)
        
        if data_availability_ds: plotting_utils.hatch(
            ax,
            data_availability_ds.to_array().squeeze(),
            **plotting_utils.no_data_plot_kwargs
        )
        toe_ds[metric].plot(cmap=cmap, ax=ax, **toe_plot_kwargs)
        
        ax.coastlines()
        plotting_utils.add_lat_markers(ax, side='right')

        ax.set_title(toe_plots.METRIC_MAP.get(metric, metric).replace('\n', ' '),fontsize=12 * fontscale)

    cbar1 = plotting_utils.create_discrete_colorbar(
        cmap, levels, cax, cbar_title,
        fontscale=fontscale, orientation='horizontal', pad=7,
        tick_slice=slice(None, None, 2)
    )


    return fig, axes, cax

def plot_time_of_emergence_maps_by_coord(
    toe_ds,
    does_not_emerge_ds,
    coord_name='member',
    coord_subset=None,
    fontscale=1.1,
    levels=np.arange(1920, 2030, 10),
    cmap=plt.cm.RdYlBu,
    toe_emergence_levels=None,
    fig=None,
    axes=None,
    cax=None,
    ncol=3,
    cbar_title='Time of Emergence',
    logginglevel='ERROR'
):
    """
    Plots ToE maps across slices along a specified coordinate (e.g. member, model, region).
    """
    utils.change_logginglevel(logginglevel)

    # Get coordinate values to loop through
    coord_vals = coord_subset if coord_subset is not None else toe_ds[coord_name].values

    if axes is None:
        nplots = len(coord_vals)
        num_rows = int(np.ceil(nplots / ncol))
        fig = plt.figure(figsize=(5 * ncol, 3 * num_rows))

        outer_gs = gridspec.GridSpec(2, 1, height_ratios=[0.97, 0.03], hspace=0.05)
        
        gs = gridspec.GridSpecFromSubplotSpec(num_rows, ncol, wspace=0.1, hspace=0.2, subplot_spec=outer_gs[0])

        total_slots = num_rows * ncol
        diff = total_slots - nplots
        positions_to_fill = list(range(nplots))

        if diff == 1:
            positions_to_fill = positions_to_fill[:-1]
            leftmost = total_slots - ncol
            rightmost = total_slots - 1
            positions_to_fill += [leftmost + 1, rightmost + 1]
        elif diff == 2:
            positions_to_fill = positions_to_fill[:-1]
            mid_position = total_slots - ncol + (ncol // 2)
            positions_to_fill.append(mid_position)

        axes = np.array([fig.add_subplot(gs[num], projection=ccrs.PlateCarree()) for num in positions_to_fill])
        cax = plt.subplot(outer_gs[-1, :])

    toe_plot_kwargs = dict(add_colorbar=False, levels=levels)

    for i, coord_val in enumerate(coord_vals):
        logger.debug(f'{i} - {coord_val}')
        ax = axes[i]
        sel_kw = {coord_name: coord_val}

        does_not_emerge_ds.sel(**sel_kw).plot(ax=ax, **no_emergence_plot_kwargs)

        toe_ds.sel(**sel_kw).plot(ax=ax, cmap=cmap, **toe_plot_kwargs)

        ax.coastlines()
        # plotting_utils.add_lat_markers(ax, side='right')
        ax.set_title(f"{coord_name}: {coord_val}", fontsize=12 * fontscale)

    cbar1 = plotting_utils.create_discrete_colorbar(
        cmap, levels, cax, cbar_title,
        fontscale=1.6, orientation='horizontal', pad=7,
        # tick_slice=slice(None, None, 1)
    )

    
    return fig, axes, cax

def add_agg_line(axes, ds, regions):
    toe_agg_df = ds.to_pandas()
    for ax, region in zip(axes, regions):
        reg_df = toe_agg_df.loc[region, :]
        for metric, loc in reg_df.items():
            # Define visual style explicitly for each metric
            label = toe_plots.METRIC_MAP.get(metric, metric)
            color = toe_plots.TEST_STYLES.get(metric, {}).get('color', 'black')
            # Define linestyle, zorder, and visibility tweaks
            if metric == 'sn':
                # linestyle = '--'
                zorder = 2
                linewidth = 2
            elif metric == 'ks':
                # linestyle = '-.'
                zorder = 3
                linewidth = 2
            elif metric == 'frac':
                # linestyle = '-'
                zorder = 4
                linewidth = 2.5  # Slightly thicker to stand out
            ax.axvline(loc, label=label, linewidth=1.5, linestyle='-', color=color)


def plot_percent_emerged_series(
    time_series_tree, variables=None, toe_metrics=None, dim=None, dim_vals = None,
    fig=None, axes=None, labels='ylabels', bbox_to_anchor=None, ncol=3, logginglevel='ERROR'
):

    def _format_label(value, dim):
        """Format label for y-axis or annotations depending on dim."""
        if dim == "region":
            return toe_const.NAMING_MAP.get(value, value) \
                .replace("Mid", "Mid-\n") \
                .replace("itudes", "itudes\n")
        else:
            return str(value)

    def _apply_labels(ax, labels, row, column, value, dim):
        if (labels == 'ylabels' and row == 4 and column == 0) or \
           (labels == 'annotations' and row == 3) or \
           (labels == 'ylabels_lhs' and column != 0 and row == 4):
            ax.set_ylabel('Percent of Surface Area Emerged', fontsize=12)

        if labels == 'ylabels':
            if column == len(variables)-1:
                ax.set_ylabel(
                    _format_label(value, dim),
                    fontsize=10, rotation=0, labelpad=25
                )
                ax.yaxis.tick_right()

        elif labels == 'annotations':
            ax.annotate(
                _format_label(value, dim),
                xy=(0.95, 0.1), xycoords='axes fraction',
                ha='right', va='center', fontsize=15
            )

        elif labels == 'ylabels_lhs':
            if column == 0:
                ax.set_ylabel(
                    _format_label(value, dim),
                    fontsize=10, rotation=0, labelpad=25,
                    ha='center', va='center'
                )

    if variables is None:
        variables = list(time_series_tree)
    if toe_metrics is None:
        toe_metrics = list(time_series_tree[variables[0]])
    if dim is None:
        # pick first non-time dim that exists
        for d in time_series_tree[variables[0]].ds.dims:
            if d not in ["time", "member"]:
                dim = d
                break

    all_dim_vals = list(time_series_tree[variables[0]][dim].values)
    if dim_vals is not None:
        dim_vals = np.intersect1d(dim_vals, all_dim_vals)
    else:
        dim_vals = all_dim_vals

    if axes is None:
        fig = plt.figure(figsize=(4*len(variables), len(dim_vals)))
        gs = gridspec.GridSpec(len(dim_vals), len(variables), hspace=.1, wspace=.25)
        all_axes = []

    for column, var in enumerate(variables):
        percent_time_ds = time_series_tree[var].ds
        axes_column = [fig.add_subplot(gs[i, column]) for i in range(len(dim_vals))] if axes is None else axes[:, column]

        for row, value in enumerate(dim_vals):
            ax = axes_column[row]
            percent_time_value_ds = percent_time_ds.sel({dim: value})
            if "member" in percent_time_value_ds.coords:
                percent_time_value_ds = percent_time_value_ds.median(dim="member")

            toe_plots.percent_emerged_series(
                percent_time_value_ds, toe_metrics, ax=ax,
                legend=False, fontscale=0.65,
                xticks=np.arange(1890, 2110, 10), logginglevel=logginglevel
            )

            ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
            ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
            ax.tick_params(axis="x", rotation=45)

            _apply_labels(ax, labels, row, column, value, dim)

            if row == 0:
                ax.xaxis.set_label_position("top")
                ax.xaxis.tick_top()
                ax.set_title(NAME_MAPPING.get(var, var))

            if row == 0 or row == len(dim_vals)-1:
                ax.set_xlabel("Year", fontsize=10)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

        if axes is None:
            all_axes.append(axes_column)

    # Legend placement unchanged
    if len(all_axes) == 2: midde_axes = -1
    else: midde_axes = int(np.ceil(len(all_axes)/2)) - 1
    if bbox_to_anchor is None: bbox_to_anchor = (1 if len(all_axes) == 2 else 0.5, -2.2)
    leg = all_axes[midde_axes][-1].legend(
        fontsize=10, loc="lower center", bbox_to_anchor=bbox_to_anchor, ncol=ncol
    )

    return fig, all_axes, leg
plot_percent_emerged_series
# def plot_percent_emerged_series(
#     time_series_tree, variables=None, toe_metrics=None, regions=None,
#     fig=None, axes=None, labels='ylabels', bbox_to_anchor=None, ncol=3, logginglevel='ERROR'):

#     def _apply_labels(ax, labels, row, column, region):
#         """Apply ylabels/annotations/ylabels_lhs logic to an axis."""
#         # Shared condition: only once instead of repeating
#         if (labels == 'ylabels' and row == 4 and column == 0) or \
#            (labels == 'annotations' and row == 3) or \
#            (labels == 'ylabels_lhs' and column != 0 and row == 4):
#             ax.set_ylabel('Percent of Surface Area Emerged', fontsize=12)

#         if labels == 'ylabels':
#             if column == len(variables)-1:
#                 ax.set_ylabel(
#                     toe_const.NAMING_MAP.get(region, region)
#                     .replace('Mid', 'Mid-\n')
#                     .replace('itudes', 'itudes\n'),
#                     fontsize=10, rotation=0, labelpad=25
#                 )
#                 ax.yaxis.tick_right()

#         elif labels == 'annotations':
#             ax.annotate(
#                 toe_const.NAMING_MAP.get(region, region),
#                 xy=(0.95, 0.1), xycoords='axes fraction',
#                 ha='right', va='center', fontsize=15
#             )

#         elif labels == 'ylabels_lhs':
#             if column == 0:
#                 ax.set_ylabel(
#                     toe_const.NAMING_MAP.get(region, region)
#                     .replace('Mid', 'Mid-\n')
#                     .replace('itudes', 'itudes\n'),
#                     fontsize=10, rotation=0, labelpad=25,
#                     ha='center', va='center'
#                 )

#     if variables is None: variables = list(time_series_tree)
#     if toe_metrics is None: toe_metrics = list(time_series_tree[variables[0]])
#     if regions is None: regions = list(time_series_tree[variables[0]].region.values)

#     if axes is None:
#         fig = plt.figure(figsize=(4*len(variables), len(regions)))
#         gs = gridspec.GridSpec(len(regions), len(variables), hspace=.1, wspace=.25)
#         all_axes = []

#     for column, var in enumerate(variables):
#         percent_time_ds = time_series_tree[var].ds
#         axes_column = [fig.add_subplot(gs[i, column]) for i in range(len(regions))] if axes is None else axes[:, column]

#         for row, region in enumerate(regions):
#             ax = axes_column[row]
#             percent_time_region_ds = percent_time_ds.sel(region=region)
#             if 'member' in list(percent_time_region_ds.coords):
#                 percent_time_region_ds = percent_time_region_ds.median(dim='member')

#             toe_plots.percent_emerged_series(
#                 percent_time_region_ds, toe_metrics, ax=ax,
#                 legend=False, fontscale=0.65, xticks=np.arange(1890, 2110, 10), logginglevel=logginglevel
#             )
#             ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
#             ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
#             ax.tick_params(axis='x', rotation=45)
#             ax.tick_params(axis='x', which='major', length=6, width=1)
#             ax.tick_params(axis='x', which='minor', length=3, width=1)

#             # All ylabel/annotation logic now one call
#             _apply_labels(ax, labels, row, column, region)

#             if row == 0:
#                 ax.xaxis.set_label_position('top')
#                 ax.xaxis.tick_top()
#                 ax.set_title(NAME_MAPPING.get(var, var))

#             if row == 0 or row == len(regions)-1:
#                 ax.set_xlabel('Year', fontsize=10)
#             else:
#                 ax.set_xlabel('')
#                 ax.set_xticklabels([])

#         if axes is None:
#             all_axes.append(axes_column)

#     if len(all_axes) == 2: midde_axes = -1
#     else: midde_axes = int(np.ceil(len(all_axes)/2)) - 1
#     if bbox_to_anchor is None: bbox_to_anchor = (1 if len(all_axes) == 2 else 0.5, -2.2)
#     leg = all_axes[midde_axes][-1].legend(
    #     fontsize=10, loc='lower center', bbox_to_anchor=bbox_to_anchor, ncol=ncol,
    # )

    # return fig, all_axes, leg





def style_ax(ax, title=None, ylabel=None, ylim=None, labelpad=0, fontsize=12):
    # Grid
    ax.grid(True, which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Labels
    ax.set_xlabel('Year', fontsize=14)
    if ylabel: ax.set_ylabel(ylabel, fontsize=fontsize, rotation=90, labelpad=labelpad)
    
    # Ticks
    ax.tick_params(labelsize=fontsize, direction='out', length=6, width=1, colors='black')
    
    if title is not None: ax.set_title(title, fontsize=fontsize+2, weight='bold', loc='center')
    
    ax.axhline(0, color='k', alpha=0.6, zorder=-100)
    # if ylim is not None: ax.set_ylim(*ylim)
    
    return ax

def create_stacked_order(ds: xr.Dataset, imember: int = None, metric=None) -> xr.DataArray:
    """
    Create a stacked and sorted DataArray of noise_ratio by lat/lon.

    Parameters
    ----------
    ds (xr.Dataset: )Input dataset containing 'noise_ratio'.
    imember (int, optional ): Index of ensemble member to select.

    Returns
    -------
    xr.DataArray: Stacked and sorted 'noise_ratio' with NaNs dropped.
    """
    if check_for_null(ds):  return

    if hasattr(ds, 'ds'): ds = ds.ds
    
    if isinstance(imember, int) and 'member' in list(ds.coords):
        ds = ds.isel(member=imember)

    if metric is not None: 
        ds = ds.sel(metric=metric)
        
    stacked_ds = ds['noise_ratio'].stack(latlon=['lat', 'lon'])
    stacked_ds = stacked_ds.sortby(stacked_ds, ascending=False)
    stacked_ds = stacked_ds.dropna(dim='latlon')
    stacked_ds = stacked_ds.persist()
    wait(stacked_ds);
    return stacked_ds.to_dataset()

def plot_nosie_series(axes, full_series_arr, pi_arr, time=None, fontsize=12, ylim=(-3, 3)):
    from matplotlib.ticker import MaxNLocator

    if time is None: time = np.arange(len(full_series_arr))
    
    base_series_arr = full_series_arr[:30]
    pi_arr = pi_arr-np.mean(pi_arr)
    plot_style = dict(color=toe_plots.color_list[0])
    
    axes[0].plot(time[:30], base_series_arr, **plot_style)
    axes[1].plot(time, full_series_arr, **plot_style)
    axes[2].plot(np.arange(len(pi_arr)), pi_arr, **plot_style)
    
    axes[0].set_title('Base Period ' + r'($\sigma$' + f'={np.std(base_series_arr):.2f}'+r'$^\circ C)$',
                     fontsize=fontsize)
    axes[1].set_title('Full Series ' + r'($\sigma$' + f'={np.std(full_series_arr):.2f}'+r'$^\circ C)$',
                     fontsize=fontsize)
    axes[2].set_title('piControl ' + r'($\sigma$' + f'={np.std(pi_arr):.2f}'+r'$^\circ C)$',
                     fontsize=fontsize)
    axes[2].set_xlim(0, len(pi_arr))

    for ax in axes:
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.axhline(0, color='k', alpha=0.7, zorder=-100)

from scipy.stats import norm

def overlap_to_sn(overlap_percent):
    '''
    Converts overlap to S/N ratio. This is useds here for plotting to match ticks.
    Formula comes from
    https://www.tandfonline.com/doi/pdf/10.1080/03610928908830127
    '''
    return -2 * norm.ppf(overlap_percent/200.0)

def plot_group(ax_anom, ax_metric, da, met_da, signal_da, frac_arr=None, color='black'):
    """
    Plot anomalies, signal, S/N metrics and overlap on paired axes.
    The S/N metrics are plotted on the left y-axis. The frac is first converted to S/N ratio
    using 'overlap_to_sn'. The function 'format_sn_axis' then creates a secondary axis to match
    the overlap, but the secondary axis is just cosmetic.

    Parameters
    ----------
    ax_anom : matplotlib.axes.Axes
        Axis for plotting the anomaly time series and the signal estimate.
    ax_metric : matplotlib.axes.Axes
        Axis for plotting S/N metrics and converted fraction values.
    da : xarray.DataArray
        Time series of anomalies to plot.
    met_da : xarray.Dataset
        Dataset containing S/N metric time series (e.g., sn_lowess_base).
    signal_da : xarray.DataArray
        Smoothed or externally defined signal estimate.
    frac_arr : array-like, optional
        Array of overlap percentages (0–100). If not provided,
        uses `met_da['frac']`.
    color : str, default 'black'
        Color for the signal line.

    Returns
    -------
    ax_metric : matplotlib.axes.Axes
        Axis with plotted S/N metrics and converted fraction values.

    """

    time = da.time.dt.year.values
    data_style = {'linewidth': 2, 'alpha': 0.85}

    # anomalies + signal
    ax_anom.plot(time, da.values, label='Anomalies', color=toe_plots.color_list[0], **data_style)
    ax_anom.plot(time, signal_da.values, label='Signal', color=color, **data_style)

    # --- plot SN metrics on left axis ---
    sn_metrics = ['sn_lowess_base', 'sn_lowess_full', 'sn_lowess_pi']
    for metric in sn_metrics:
        style = toe_plots.TEST_STYLES.get(metric, {}, drop_keys='marker')
        if metric not in list(met_da):
            continue
        ax_metric.plot(
            time, met_da[metric].values,
            label=toe_plots.METRIC_MAP[metric],
            linewidth=2, **style
        )

    # --- plot frac on left axis, converted to S/N ---
    frac_style = toe_plots.TEST_STYLES.get('frac', {}, drop_keys='marker')
    if frac_arr is None: frac_arr = met_da['frac'].values
    sn_from_frac = overlap_to_sn(frac_arr)   # convert %
    sn_from_frac = np.minimum(sn_from_frac, 6)
    
    ax_metric.plot(
        time, sn_from_frac,
        label=toe_plots.METRIC_MAP['frac'],
        linewidth=2, **frac_style
    )

    return ax_metric



def format_sn_axis(
    ax: plt.axes,
    sn_ticks: List[float],
    sn_labels: Optional[List[str | float]],
    overlap_ticks: List[float],
    overlap_labels: Optional[List[str]] = None,
    fontsize=12, 
) -> plt.axes:
    """
    Apply S/N ticks and add a secondary axis with overlap ticks/labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to format.
    sn_ticks, sn_labels : lists
        Tick positions and optional labels for S/N axis (use ticks if labels is None).
    overlap_ticks, overlap_labels : lists
        Tick positions and optional labels for overlap axis (use ticks if labels is None).
    ylim : tuple, default (0, 15)
        Y-axis limits.

    Returns
    -------
    ax, ax_frac : matplotlib.axes.Axes
        Primary and secondary axes.
    """
    ax.set_yscale("symlog")#, linthresh=6)
    # ax.set_ylim(ylim) 
    ax.set_yticks(sn_ticks)
    ax.set_ylim(np.take(sn_ticks, [0, -1]))
    ax.set_yticklabels(sn_labels if sn_labels is not None else sn_ticks, fontsize=fontsize)
    ax.grid(True, alpha=0.6)

    ax_frac = ax.twinx()
    ax_frac.set_ylim(ax.get_ylim())
    ax_frac.set_yscale(ax.get_yscale())#"symlog", linthresh=6)
    ax_frac.set_yticks(overlap_ticks)
    ax_frac.set_yticklabels(overlap_labels if overlap_labels is not None else overlap_ticks,
                           fontsize=fontsize)
    ax_frac.set_ylabel("Overlap (%)", fontsize=14)
    ax.set_ylabel("S/N Ratio", fontsize=14)

    return ax_frac


def color_plot_group(ax_metric, ax_frac):

    # pick one SN style for the axis color (e.g. base)
    sn_style = toe_plots.TEST_STYLES['sn_lowess_base']
    ax_metric.spines['left'].set_color(sn_style['color'])
    ax_metric.yaxis.label.set_color(sn_style['color'])
    ax_metric.tick_params(axis='y', colors=sn_style['color'])

    # --- plot frac on right axis ---
    frac_style = toe_plots.TEST_STYLES['frac']

    ax_frac.spines['right'].set_color(frac_style['color'])
    ax_frac.yaxis.label.set_color(frac_style['color'])
    ax_frac.tick_params(axis='y', colors=frac_style['color'])



def style_hexbin_axes(axes, ylim=(-60, 60), xlim=(0.1, 6), fontsize=16):
    """Apply consistent style for hexbin panels."""
    for ax in axes:
        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)
        ax.grid(True, linestyle="--", color="grey", alpha=0.5)
        ax.axhline(0, color="k", zorder=-1000)
        ax.axvline(1, color="k", zorder=-1000)
        ax.set_title(None)
        ax.set_xlabel("Relative Noise Change", fontsize=fontsize)
        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)



def flatten_and_remove_nans(toe_diff_ds, noise_change_ds):
    '''
    Flatten values and remove nans
    '''
    toe_diff_flat = toe_diff_ds.values.flatten()
    noise_change_flat = np.abs(noise_change_ds.values.flatten())

    mask = np.isfinite(noise_change_flat) & np.isfinite(toe_diff_flat)
    noise_change_flat = noise_change_flat[mask]
    toe_diff_flat = toe_diff_flat[mask]

    return toe_diff_flat, noise_change_flat
