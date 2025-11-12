import os
import sys
from typing import Dict

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

import utils
from utils import logger

sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))
import plotting_utils
import toe_constants as toe_const

color_list = ['#1f77b4', '#ff7f0e', '#8bc34a']

main_toe_metrics = ('sn_lowess_base', 'ks', 'ttest', 'frac', 'hd') #, 'perkins'
main_toe_metrics_no_hype = ('sn_lowess_base', 'frac', 'hd')
regions = np.array(['global', 'tropics','arctic', 'antarctic', 'land', 'ocean'])

# regions = np.array(['global', 'land', 'ocean',  'antarctic', 'mid_lat_sh', 'tropics', 'mid_lat_nh', 'arctic',])

metrics_dict = {
    "all": np.array([
        'sn_lowess_base', 'sn_lowess_full', 'sn_lowess_pi',
        'sn_lowess_roll', 'sn_mean_base', 'sn_mean_pi', 'sn_mean_roll',
        'sn_poly4_base', 'sn_poly_pi', 'sn_ens_med_base', 'sn_ens_med_pi', 'ks', 'ks_bbs',
        'ttest', 'frac', 'hd',
    ]),
    "main": np.array([
        'sn_lowess_base', 'sn_lowess_full', 'sn_lowess_pi',
        'sn_mean_base', 'sn_poly4_base', 'frac', 'ks', 'ks_bbs', 'ttest', 'ttest_bbs'
    ]),
    "supplementary": np.array([
        'sn_lowess_base', 'sn_lowess_full', 'sn_lowess_pi', 'sn_lowess_roll',
        'sn_mean_base', 'sn_poly4_base', 'sn_ens_med_base',
        'frac', 'hd', 'ks', 'ks_bbs', 'ttest', 'ttest_bbs', 'mwu'
    ]),
    'hype': np.array(['ks_window', 'ks_bbs_window', 'ttest_window', 'ttest_bbs_windwo', 'mwu']), 
    "all_no_hype": np.array([
        'sn_lowess_base', 'sn_lowess_full', 'sn_lowess_pi',
        'sn_lowess_roll', 'sn_mean_base', 'sn_mean_pi', 'sn_mean_roll',
        'sn_poly4_base', 'sn_poly_pi', 'sn_ens_med_base', 'sn_ens_med_pi',
        'frac', 'hd',
    ]), 
    "main_no_hype": np.array([
        'sn_lowess_base', 'sn_lowess_full', 'sn_lowess_pi',
        'sn_mean_base', 'sn_poly4_base', 'frac'
        # 'ks', 'ttest' removed
    ]),
    "supplementary_no_hype": np.array([
        'sn_lowess_base', 'sn_lowess_full', 'sn_lowess_pi', 'sn_lowess_roll',
        'sn_mean_base', 'sn_poly4_base', 'sn_ens_med_base',
        'frac', 'hd'
        # 'ks', 'ttest' removed
    ])
}



NAME_MAPPING = {
    'best_tas': 'Surface Temperature\n(Berkeley Earth)',
    'access_ssp585_tas': 'Surface Temperature\n(ACCESS-ESM1-5)',
    'access_ssp585_pr': 'Annual Precipitation\n(ACCESS-ESM1-5)'
}

METRIC_NAME_MAP = {
    # --- S/N ---
    'sn': r'S/N$_{\mathrm{LOWESS,\ base}}$ Ratio',
    'sn_lowess_base': r'S/N$_{\mathrm{LOWESS,\ base}}$ Ratio',
    'sn_lowess_full': r'S/N$_{\mathrm{LOWESS,\ full}}$ Ratio',
    'sn_lowess_pi': r'S/N$_{\mathrm{LOWESS,\ piC.}}$ Ratio',
    'sn_lowess_roll': r'S/N$_{\mathrm{LOWESS,\ roll}}$ Ratio',
    'sn_mean': r'S/N$_{\mathrm{mean,\ base}}$ Ratio',
    'sn_mean_base': r'S/N$_{\mathrm{mean,\ base}}$ Ratio',
    'sn_mean_pi': r'S/N$_{\mathrm{mean,\ piC.}}$ Ratio',
    'sn_mean_roll': r'S/N$_{\mathrm{mean,\ roll}}$ Ratio',
    'sn_poly4_base': r'S/N$_{\mathrm{poly4,\ base}}$ Ratio',
    'sn_poly_pi': r'S/N$_{\mathrm{poly4,\ piC.}}$ Ratio',
    'sn_ens_med': r'S/N$_{\mathrm{Ens.\ mean}}$ Ratio',
    'sn_ens_med_base': r'S/N$_{\mathrm{Ens.\ mean,\ base}}$ Ratio',
    'sn_ens_med_pi': r'S/N$_{\mathrm{Ens.\ mean,\ piC.}}$ Ratio',

    # --- Statistical tests ---
    'ks': 'Kolmogorov-\nSmirnov Test',
    'ks_bbs': 'Kolmogorov-\nSmirnov Test \n(Block Bootstrap)',
    'ttest': 'T-Test',
    'ttest_bbs': 'T-Test\n(Block Bootstrap)',
    'mwu': 'Mann–Whitney\nU Test',

    # --- Distances / overlap ---
    'perkins': 'Perkins\nSkill Score',
    'frac': 'Area of\nOverlap',
    'hd': 'Hellinger\nDistance'
}

METRIC_NAME_MAP['ks_bbs_window'] = METRIC_NAME_MAP['ks_bbs']
METRIC_NAME_MAP['ks_window'] = METRIC_NAME_MAP['ks']
METRIC_NAME_MAP['ttest_bbs_window'] = METRIC_NAME_MAP['ttest_bbs']
METRIC_NAME_MAP['ttest_window'] = METRIC_NAME_MAP['ttest']


METRIC_NAME_MAP_SHORT = {
    # --- S/N LOWESS ---
    'sn': r'S/N$_{\mathrm{LOWESS,\ base}}$',
    'sn_lowess_base': r'S/N$_{\mathrm{LOWESS,\ base}}$',
    'sn_lowess_full': r'S/N$_{\mathrm{LOWESS,\ full}}$',
    'sn_lowess_pi': r'S/N$_{\mathrm{LOWESS,\ piC}}$',
    'sn_lowess_roll': r'S/N$_{\mathrm{LOWESS,\ roll}}$',

    # --- S/N mean ---
    'sn_mean': r'S/N$_{\mathrm{mean,\ base}}$',
    'sn_mean_base': r'S/N$_{\mathrm{mean,\ base}}$',
    'sn_mean_pi': r'S/N$_{\mathrm{mean,\ piC}}$',
    'sn_mean_roll': r'S/N$_{\mathrm{mean,\ roll}}$',

    # --- S/N poly4 ---
    'sn_poly4_base': r'S/N$_{\mathrm{poly4,\ base}}$',
    'sn_poly_pi': r'S/N$_{\mathrm{poly4,\ piC}}$',

    # --- S/N ensemble median ---
    'sn_ens_med': r'S/N$_{\mathrm{Ens.\ mean}}$',
    'sn_ens_med_base': r'S/N$_{\mathrm{Ens.\ mean,\ base}}$',
    'sn_ens_med_pi': r'S/N$_{\mathrm{Ens.\ mean,\ piC}}$',

    # --- Stats tests ---
    'ks': 'KS test',
    'ks_bbs': 'KS test(BBS)',
    'ttest': 'T-test',
    'ttest_bbs': 'T-test (BBS)',
    'mwu': 'MWU',

    # --- Distances / overlap ---
    'perkins': 'PSS',
    'frac': 'AO',
    'hd': 'HD'
}

METRIC_NAME_MAP_SHORT['ks_bbs_window'] = METRIC_NAME_MAP_SHORT['ks_bbs']
METRIC_NAME_MAP_SHORT['ks_window'] = METRIC_NAME_MAP_SHORT['ks']
METRIC_NAME_MAP_SHORT['ttest_bbs_window'] = METRIC_NAME_MAP_SHORT['ttest_bbs']
METRIC_NAME_MAP_SHORT['ttest_window'] = METRIC_NAME_MAP_SHORT['ttest']


METRIC_MAP = METRIC_NAME_MAP
METRIC_MAP_SHORT = METRIC_NAME_MAP_SHORT

class StyleDict(dict):
    def get(self, key, default=None, *, drop_keys=None):
        """
        Get a style dict with optional keys removed.

        Parameters
        ----------
        key : str
            The style key (e.g., 'ks', 'sn_lowess_base').
        default : dict, optional
            Default dict if key not found.
        drop_keys : list[str] or None
            Keys to exclude from the returned dict.
            If None, returns everything.
        """
        d = super().get(key, default)
        if isinstance(d, dict) and drop_keys:
            return {k: v for k, v in d.items() if k not in drop_keys}
        return d


TEST_STYLES = StyleDict({
    # --- S/N LOWESS (anchored to color_list[0]) ---
    'sn_lowess_base': {'color': color_list[0], 'linestyle': 'solid',  'marker': 'o'},
    'sn_lowess_full': {'color': color_list[0], 'linestyle': 'dashed', 'marker': 's'},
    'sn_lowess_pi':   {'color': color_list[0], 'linestyle': 'dotted', 'marker': 'D'},
    'sn_lowess_roll': {'color': color_list[0], 'linestyle': 'dashdot','marker': '^'},

    # --- S/N MEAN (greys) ---
    'sn_mean_base': {'color': '#4D4D4D', 'linestyle': 'solid',  'marker': 'v'},
    'sn_mean_pi':   {'color': '#4D4D4D', 'linestyle': 'dashed', 'marker': 'P'},
    'sn_mean_roll': {'color': '#4D4D4D', 'linestyle': 'dotted', 'marker': 'X'},

    # --- S/N POLY4 (browns) ---
    'sn_poly4_base': {'color': '#8C510A', 'linestyle': 'solid',  'marker': '*'},
    'sn_poly_pi':    {'color': '#8C510A', 'linestyle': 'dashed', 'marker': 'h'},

    # --- S/N ENS_MED (purples) ---
    'sn_ens_med_base': {'color': '#762A83', 'linestyle': 'solid',  'marker': 'p'},
    'sn_ens_med_pi':   {'color': '#762A83', 'linestyle': 'dashed', 'marker': '<'},

    # --- Statistical tests ---
    'ks':        {'color': color_list[1], 'linestyle': 'solid',   'marker': 'o'},  # orange base
    'ks_bbs':    {'color': color_list[1], 'linestyle': 'dotted',  'marker': 's'},  # orange bootstrap
    
    'ttest':     {'color': '#d73027',     'linestyle': 'solid',   'marker': 'v'},  # red base
    'ttest_bbs': {'color': '#d73027',     'linestyle': 'dashed',  'marker': 'P'},  # red bootstrap
    
    'mwu':       {'color': '#542788',     'linestyle': 'solid',   'marker': 'D'},  # violet



    # --- Distances / overlap (greens, varied shades) ---
    'frac':    {'color': '#8bc34a', 'linestyle': 'solid',  'marker': 'o'},  # bright green
    'perkins': {'color': '#4caf50', 'linestyle': 'dashed', 'marker': 's'},  # darker green
    'hd':      {'color': '#c7e9c0', 'linestyle': 'dotted', 'marker': 'D'}   # pale mint green
}
                       )

TEST_STYLES['ks_bbs_window'] = TEST_STYLES['ks_bbs']
TEST_STYLES['ks_window'] = TEST_STYLES['ks']

TEST_STYLES['ttest_bbs_window'] = TEST_STYLES['ttest_bbs']
TEST_STYLES['ttest_window'] = TEST_STYLES['ttest']

# Alias so you can use either name
TEST_PLOT_DICT = TEST_STYLES
test_styles = TEST_STYLES


# TEST_PLOT_DICT = {
#     # --- S/N family (base = blue #1f77b4) ---
#     'sn_lowess_base':   {'color': '#1f77b4', 'marker': 'o', 'linestyle': 'solid'},
#     'sn_lowess_full':   {'color': '#2a82c8', 'marker': 's', 'linestyle': 'dashed'},
#     'sn_lowess_pi':     {'color': '#3390dc', 'marker': 'D', 'linestyle': 'dotted'},
#     'sn_lowess_roll':   {'color': '#4da3e7', 'marker': '^', 'linestyle': 'dashdot'},
#     'sn_mean_base':     {'color': '#66b2ff', 'marker': 'v', 'linestyle': 'solid'},
#     'sn_mean_pi':       {'color': '#80c1ff', 'marker': 'P', 'linestyle': 'dashed'},
#     'sn_mean_roll':     {'color': '#99cfff', 'marker': 'X', 'linestyle': 'dotted'},
#     'sn_poly4_base':    {'color': '#b3ddff', 'marker': '*', 'linestyle': 'solid'},
#     'sn_poly_pi':       {'color': '#cceaff', 'marker': 'h', 'linestyle': 'dashed'},
#     'sn_ens_med_base':  {'color': '#e6f5ff', 'marker': 'p', 'linestyle': 'solid'},
#     'sn_ens_med_pi':    {'color': '#f2faff', 'marker': '<', 'linestyle': 'dashed'},

#     # --- Statistical tests family (base = orange #ff7f0e) ---
#     'ks':        {'color': '#ff7f0e', 'marker': 'o', 'linestyle': 'solid'},   # base
#     'ks_bbs':    {'color': '#ff9933', 'marker': 's', 'linestyle': 'dotted'},  # bootstrap
#     'ttest':     {'color': '#ff7f0e', 'marker': 'v', 'linestyle': 'dashed'},  # base
#     'ttest_bbs': {'color': '#ff9933', 'marker': 'P', 'linestyle': 'dashdot'}, # bootstrap
#     'mwu':       {'color': '#ffd9b3', 'marker': 'h', 'linestyle': (0, (3, 5, 1, 5))},  # MWU custom

#     # --- Distances / overlap family (base = green #8bc34a) ---
#     'frac':    {'color': '#8bc34a', 'marker': 'o', 'linestyle': 'solid'},
#     'perkins': {'color': '#a6d96a', 'marker': 's', 'linestyle': 'dashed'},
#     'hd':      {'color': '#c7e9b4', 'marker': 'D', 'linestyle': 'dotted'}
# }


test_styles = TEST_STYLES




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



def percent_emerged_series(
    emergence_series_da, toe_metric_list: np.ndarray = None,
    xticks=None, 
    time=None, ax=None, legend=True, fontscale=1, logginglevel='ERROR'):
    """
    Plots percent emerged series for specified metrics.

    Parameters:
    - emergence_series_da: xarray.DataArray containing emergence data for different metrics.
    - toe_metric_list: List or array of metrics to plot. Defaults to all metrics in the DataArray.
    - time: Time values to use for plotting. Defaults to `emergence_series_da.time.values`.
    - fig: Matplotlib figure object. If None, a new figure and axis are created.

    Returns:
    - None
    """
    utils.change_logginglevel(logginglevel)
    # Create figure and axis if not provided
    if ax is None: fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Use default time and metrics if not provided
    if time is None:
        if hasattr(emergence_series_da.time, "dt"):
            time = emergence_series_da.time.dt.year.values
        else:
            time = emergence_series_da.time.values

    
    toe_metric_list = list(emergence_series_da) if toe_metric_list is None else toe_metric_list
    # Incase any provided that are not actually in the data
    toe_metric_list = [tm for tm in toe_metric_list if tm in list(emergence_series_da)]

    # Loop through metrics and plot
    logger.trace(emergence_series_da)

    for i, metric in enumerate(toe_metric_list):
        # Get the color and label for the metric
        style = TEST_STYLES.get(metric, {'color': 'black'}, drop_keys=['marker'])
        #TEST_PLOT_DICT.get(metric, {'color': 'black'}) # TEST_PLOT_DICT, TEST_STYLES
        # color = test_colors.get(metric, 'black')  # Default to black if not in the dictionary
        label = METRIC_MAP.get(metric, metric)  # Fallback to metric name if no conversion
        logger.debug(f'\n{label=}, {style=}')
        # Plot the data

        if 'member' in list(emergence_series_da.coords):
            ax.plot(
                time, 
                emergence_series_da[metric].median(dim='member').squeeze().values, 
                label=label, 
                linewidth=3,
                **style
            )
    
            ax.fill_between(
                time, 
                emergence_series_da[metric].quantile(0.10, dim='member').squeeze().values, 
                emergence_series_da[metric].quantile(0.90, dim='member').squeeze().values, 
                label=label,  linewidth=3, alpha=0.5, **style)

        else:
            ax.plot(
                time, 
                emergence_series_da[metric].squeeze().values, 
                label=label,  linewidth=3, zorder = len(toe_metric_list) - i, **style)

    # Customize the plot
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='y', labelsize=12*fontscale)
    ax.tick_params(axis='x', labelsize=12*fontscale)
    ax.set_yticks(np.arange(0, 120, 20))
    ax.set_ylim(-2, 102)
    ax.set_xlim(np.take(xticks, [0, -1]))
    if legend: ax.legend(fontsize=14*fontscale, loc='upper left')
    if legend: ax.legend(fontsize=14*fontscale, loc='upper left')
        
    # if xticks is not None:
    #     ax.set_xticks(xticks)
    #     xticks_labels = xticks.astype(str)
    #     xticks_labels[::2] = ''
    #     ax.set_xticklabels(xticks_labels)
