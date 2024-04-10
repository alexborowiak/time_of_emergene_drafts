import os
import sys
from typing import Dict

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))
import plotting_utils
import toe_constants as toe_const

TEST_PLOT_DICT = {
    'ks': {'color': 'red', 'marker': 'o'},
    'ttest': {'color': 'orange', 'marker': 'x'},
    'anderson': {'color': 'sienna', 'marker': '^'},
    'signal_to_noise': {'color': 'blue', 'marker': 'o'}
}


def flip_pvalue(pvalue:float):
    '''
    The p-value is often best to plot with 0 towards the top, however, the axes often erros and doesn't allow the flip.
    Thus, flipping the value manually can bve needed
    '''
    return np.abs(pvalue-1)

def plot_multiseries_with_pvalues(
    series_ds: xr.Dataset,
    exceedance_year_ds: xr.Dataset,
    best_ds_smean: xr.Dataset,
    labels:Dict[str, str] = {},
    fig=None,
    axes=None,
    gs=None
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
    fig = plt.figure(figsize=(12, 8)) if fig is None else fig
    gs = gridspec.GridSpec(3, 1, hspace=0) if gs is None else gs

    # Moved the best raw data to the top
    ax1, ax2, ax3 = [fig.add_subplot(gs[i]) for i in range(3)] if axes is None else axes

    legend_lines = []  # For storing lines for the legend

    ax1.plot(best_ds_smean.time.values, best_ds_smean.values, color='black', alpha=0.7)

    time = series_ds.time.values
    for series_name in series_ds.data_vars:
        series_data = series_ds[series_name]
        if series_name in toe_const.PVALUE_TESTS:
            ax = ax3
            series_vals = flip_pvalue(series_data.to_numpy())
        elif series_name == 'signal_to_noise':
            ax = ax2
            series_vals = series_data

        color = TEST_PLOT_DICT[series_name]['color']
        label = toe_const.NAME_CONVERSION_DICT.get(series_name, series_name)
        ax.plot(time, series_vals, c=color, label=label)
        if series_name in toe_const.PVALUE_TESTS:
            legend_lines.append(ax.plot([], [], color=color, label=label)[0])

    for ax, color in zip([ax3, ax2], ['red', 'blue']):
        ax.spines['left'].set_color(color)
        ax.tick_params(axis='y', color=color, labelcolor=color)

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(*np.take(best_ds_smean.time.values, [0, -1]))

    
    for test_name in exceedance_year_ds.data_vars:
        # Get the year of exceedance
        year_of_emergence_int = int(exceedance_year_ds[test_name].values)
        # Select the series at the year
        series_year_select = series_ds[test_name].sel(time=series_ds.time.dt.year == year_of_emergence_int)
        # Get the actual time stamp
        year_of_emergence = series_year_select.time.values
        # Get the y-value
        val = float(series_year_select.values)
        if test_name in toe_const.PVALUE_TESTS:
            val = flip_pvalue(val)
        color = TEST_PLOT_DICT[test_name]['color']  # Get color from dict or default to black
        ax = ax2 if test_name == 'signal_to_noise' else ax3
        ax.scatter(year_of_emergence, val, color=color, marker=TEST_PLOT_DICT[test_name]['marker'], s=65)

    ax3.set_ylabel('p-value', color='red')
    ax2.set_ylabel('Signal-to-Noise Ratio', color='blue')
    ax1.set_ylabel('Surface Temperature\nAnomaly (K)' if 'ylabel_bottom' not in labels else labels['ylabel_bottom'])
    ax1.set_xlabel('Year')

    ax3.set_yticklabels([label.get_text() for label in ax3.get_yticklabels()][::-1])


    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    # Create the legend
    legend = ax3.legend(ncol=1, handles=legend_lines, loc='center', bbox_to_anchor=(0.8, 0.5), frameon=True, fontsize=12)
    frame = legend.get_frame()
    frame.set_color('white')  # Set the legend frame color to white
    frame.set_edgecolor('black')  # Set the legend frame edge color to black
    ax3.add_artist(legend)





