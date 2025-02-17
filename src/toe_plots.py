import os
import sys
from typing import Dict

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))
import plotting_utils
import toe_constants as toe_const

TEST_PLOT_DICT = {
    'sn': {'color': 'green', 'marker': 'o'},
    'sn_lowess': {'color': 'green', 'marker': 'o'},
    'sn_poly4': {'color': 'blue', 'marker': 'o'},
    'sn_rolling': {'color': 'purple', 'marker': 'o'},
    'sn_anom': {'color': 'darkorchid', 'marker': 'o'},
    'nn': {'color': 'yellow', 'marker': 'o'},


    'sn_lowess_rolling': {'color': 'darkgreen', 'marker': 'o'},
    'sn_lowess_rolling_smooth': {'color': 'green', 'marker': 'o'},
    
    'ks': {'color': 'red', 'marker': '^'},
    'ttest': {'color': 'darkorange', 'marker': '^'},
    'anderson': {'color': 'sienna', 'marker': '^'},

    'frac': {'color': 'violet', ',marker':'x'},
    'perkins': {'color':'dimgrey', 'marker':'x'},
    'hd': {'color':'purple', 'marker':'x'}
}



def format_lat_lon_title(location):
    lat = location['lat']
    lon = location['lon']
    
    lat_direction = 'N' if lat >= 0 else 'S'
    lon_direction = 'E' if lon >= 0 else 'W'
    
    lat_str = f"{abs(lat)}° {lat_direction}"
    lon_str = f"{abs(lon)}° {lon_direction}"
    
    return f"{lat_str}, {lon_str}"



def flip_value(pvalue:float, flip_around:float=1):
    '''
    The p-value is often best to plot with 0 towards the top, however, the axes often erros and doesn't allow the flip.
    Thus, flipping the value manually can bve needed
    '''
    return np.abs(pvalue-flip_around)

def plot_multiseries_with_pvalues(
    series_ds: xr.Dataset,
    exceedance_year_ds: xr.Dataset,
    best_ds_smean: xr.Dataset,
    labels:Dict[str, str] = {},
    fig=None,
    axes=None,
    gs=None,
    return_figure=False,
):
    """
    Plot multiple time series with p-values and year of exceedance markers.

    Parameters:
        series_ds (xr.Dataset): Dataset containing time series data.
        exceedance_year_ds (xr.Dataset): Dataset containing years of exceedance.
        best_ds_smean (xr.Dataset): Dataset containing best data for plotting.
        fig (matplotlib.figure.Figure, optional): Figure to use for plotting.
        axes (list of matplotlib.axes.Axes, optional): List of Axes to plot on.
        gs (matplotlib.gridspec.GridSpec, optional): GridSpec for the figure.

    Returns:
        None
    """
    fig = plt.figure(figsize=(12, 15)) if fig is None else fig
    gs = gridspec.GridSpec(4, 1, hspace=0.2) if gs is None else gs

    # Moved the best raw data to the top
    ax1, ax2, ax3, ax4 = [fig.add_subplot(gs[i]) for i in range(4)] if axes is None else axes
    axes = [ax1, ax2, ax3, ax4]


    ax1.plot(best_ds_smean.time.values, best_ds_smean.values, color='black', alpha=0.7)

    time = series_ds.time.values
    legend_lines_pvalue = []  # For storing lines for the legend
    legend_lines_overlap = []  # For storing lines for the legend

    for series_name in series_ds.data_vars:
        series_data = series_ds[series_name]

        
        if 'sn' in series_name:
            ax = ax2
            # series_vals = series_data
        elif series_name in toe_const.PVALUE_TESTS:
            ax = ax3
            # series_vals = flip_value(series_data.to_numpy())

        elif series_name in toe_const.OVERLAP_TESTS:
            ax = ax4
            # series_vals = threshold = flip_value(series_data, 100)
        series_vals = series_data
        color = TEST_PLOT_DICT[series_name]['color']
        label = toe_const.NAME_CONVERSION_DICT.get(series_name, series_name)
        
        ax.plot(time, series_vals, c=color, label=label)
        
        if series_name in toe_const.PVALUE_TESTS:
            legend_lines_pvalue.append(ax.plot([], [], color=color, label=label)[0])
        if series_name in toe_const.OVERLAP_TESTS:
            legend_lines_overlap.append(ax.plot([], [], color=color, label=label)[0]) 

        threshold = toe_const.EMERGENCE_THRESHOLD_DICT.get(series_name, None)
        if threshold:
            # if series_name in toe_const.PVALUE_TESTS: threshold = flip_value(threshold)
            # elif series_name in toe_const.OVERLAP_TESTS: threshold = flip_value(threshold, 100)
            ax.axhline(threshold, color=color, linestyle='--', alpha=0.7)

    ax2.spines['left'].set_color(TEST_PLOT_DICT['sn']['color'])
    ax3.spines['left'].set_color(TEST_PLOT_DICT['ks']['color'])
    ax4.spines['left'].set_color(TEST_PLOT_DICT['frac']['color'])

    
    ax2.tick_params(axis='y', which='both', labelcolor=TEST_PLOT_DICT['sn']['color'])
    ax3.tick_params(axis='y', which='both', labelcolor=TEST_PLOT_DICT['ks']['color'])
    ax4.tick_params(axis='y', which='both', labelcolor=TEST_PLOT_DICT['frac']['color'])


    # Scatter the year of emergence onto plot
    for test_name in exceedance_year_ds.data_vars:
        # Get the year of exceedance
        year_of_emergence_int = int(exceedance_year_ds[test_name].values)
        # Select the series at the year
        series_year_select = series_ds[test_name].sel(time=series_ds.time.dt.year == year_of_emergence_int)
        # Get the actual time stamp
        year_of_emergence = series_year_select.time.values
        # Get the y-value
        val = float(series_year_select.values)
        

        
        color = TEST_PLOT_DICT[test_name].get('color', 'k')  # Get color from dict or default to black
        if 'sn' in test_name:ax = ax2
        elif test_name in toe_const.PVALUE_TESTS: ax = ax3
        elif test_name in toe_const.OVERLAP_TESTS: ax = ax4
            
        ax.scatter(year_of_emergence, val, color=color, 
                   marker=TEST_PLOT_DICT[test_name].get('marker', 'o'), 
                   s=65)

    ax4.set_ylabel('Percent Overlap (%)', fontsize=18, color=TEST_PLOT_DICT['frac']['color'])
    ax4.set_ylim(105, -5)
    # Flip the y-axis limits
    ax3.set_ylabel('p-value', color='red', fontsize=18)
    ax3.set_ylim(1.05, -0.05)
    ax2.set_ylabel('Signal-to-Noise Ratio', color='blue', fontsize=18)
    ax1.set_ylabel(
        'Surface Temperature\nAnomaly (K)' if 'ylabel_bottom' not in labels else labels['ylabel_bottom']
        , fontsize=18)
    
    ax4.set_xlabel('Year')


    list(map(lambda ax: ax.grid(True, linestyle='--', alpha=0.65), axes)) # Add the grid 
    list(map(lambda ax: ax.set_xlim(*np.take(time, [0, -1])), axes)) # Set the lims

    for ax in axes:
        tick_locations = list(filter(lambda t: t.year % 10 == 0, time))
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(list(map(lambda t:t.year, tick_locations)))
    
    # Create the legend
    legend = ax3.legend(ncol=1, handles=legend_lines_pvalue, loc='center', bbox_to_anchor=(0.8, 0.5), 
                        frameon=True, fontsize=18)
    frame = legend.get_frame()
    frame.set_color('white')  # Set the legend frame color to white
    frame.set_edgecolor('black')  # Set the legend frame edge color to black
    ax3.add_artist(legend)

    legend_overlap = ax4.legend(ncol=1, handles=legend_lines_overlap, loc='center', bbox_to_anchor=(0.3, 0.8), 
                        frameon=True, fontsize=18)
    frame2 = legend_overlap.get_frame()
    frame2.set_color('white')  # Set the legend frame color to white
    frame2.set_edgecolor('black')  # Set the legend frame edge color to black
    ax4.add_artist(legend_overlap)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    if return_figure: return [fig, gs, [ax1, ax2, ax3, ax4]]



def generate_custom_colormap(levels, cmap_name, range_start, range_end):
    """
    Generate a custom colormap for a given level range.

    Parameters:
    levels (array): The levels array for which to generate the colormap.
    cmap_name (str): The name of the base colormap.
    range_start (float): The starting point of the range (between 0 and 1).
    range_end (float): The ending point of the range (between 0 and 1).

    Returns:
    color_list (array): The custom colormap.
    """
    cmap = plt.get_cmap(cmap_name)
    color_list = cmap(np.linspace(0, 1, len(levels) * 3))
    color_list = color_list[int(len(color_list) * range_start):int(len(color_list) * range_end)]
    return color_list

def generate_sn_colormap_and_levels(vmin=-0.4, middle_vmax=1, upper_vmax=2, extreme_vmax=2, step=0.2):
    """
    Generate a custom colormap and levels for the SN plot.

    Returns:
    levels (array): The levels array.
    cmap (LinearSegmentedColormap): The custom colormap.
    """
    lower_levels = np.arange(vmin, step/2, step)
    middle_levels = np.arange(0, middle_vmax+step, step)
    upper_levels = np.arange(1, upper_vmax, step)
    extreme_levels = np.arange(2, extreme_vmax+2*step, step)
    levels = np.unique(np.concatenate([lower_levels, middle_levels, upper_levels, extreme_levels]))
    
    # Generate custom colormaps for each level range
    lower_cmap = generate_custom_colormap(lower_levels, 'Blues_r', 1/3, 2/3)
    middle_cmap = generate_custom_colormap(middle_levels, 'YlOrBr', 1/9, 4/9)
    upper_cmap = generate_custom_colormap(upper_levels, 'Reds', 4/9, 7/9)
    extreme_cmap = generate_custom_colormap(extreme_levels, 'cool', 6/9, 9/9)
    
    # Concatenate the color lists
    full_colorlist =  np.concatenate([lower_cmap, middle_cmap, upper_cmap, extreme_cmap])
    my_cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap", full_colorlist)
    return levels, my_cmap

def generate_overlap_colormap_and_levels():
    """
    Generate a custom colormap and levels for the overlap plot.

    Returns:
    levels (array): The levels array.
    cmap (LinearSegmentedColormap): The custom colormap.
    """
    middle_levels = np.arange(100, 62, -4)
    upper_levels = np.arange(62, 32, -4)
    extreme_levels = np.arange(32, 13, -4)
    levels = np.unique(np.concatenate([middle_levels, upper_levels, extreme_levels]))
    
    # Generate custom colormaps for each level range
    middle_cmap = generate_custom_colormap(middle_levels, 'YlOrBr', 1/9, 4/9)
    upper_cmap = generate_custom_colormap(upper_levels, 'Reds', 4/9, 7/9)
    extreme_cmap = generate_custom_colormap(extreme_levels, 'cool', 6/9, 9/9)
    
    # Concatenate the color lists
    full_colorlist = np.concatenate([middle_cmap,upper_cmap,extreme_cmap])
    my_cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap", full_colorlist[::-1])
    return levels, my_cmap


def plot_condition(ds, ax, left_column, right_column, **kwargs):
    """
    Plot a condition based on the sum of two columns in a DataArray.

    Parameters:
    ds (xr.DataArray): The DataArray to plot
    ax (matplotlib.axes.Axes): The axis to plot on
    left_column (str): The name of the left column to use in the condition
    right_column (str): The name of the right column to use in the condition

    Returns:
    None
    """
    xr.where((ds[left_column] + ds[right_column])>0, 1, 0).plot(ax=ax, **kwargs)