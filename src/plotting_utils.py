import numpy as np
from typing import List, NamedTuple
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import sys

import utils
from utils import logger


not_stable_kwargs = dict(hatches=['', '////'], alpha=0, colors=None)
no_data_plot_kwargs = dict(hatches=['', '////'], alpha=0, colors=None)


class PlotConfig(NamedTuple):
    title_size = 20
    label_size = 16
    cmap_title_size = 16
    legend_title_size = 16
    tick_size = 14
    legend_text_size = 14

def format_latlon(location):
    """
    Create a formatted plot title using latitude and longitude.

    Parameters
    ----------
    location : dict or list/tuple
        If dict: must contain 'lat' and 'lon' keys.
        If list/tuple: must be ordered [lat, lon].

    Returns
    -------
    str
        Formatted title in the form "{latitude}°{N/S}, {longitude}°{E/W}".
    """
    if isinstance(location, dict):
        lat, lon = location['lat'], location['lon']
    elif isinstance(location, (list, tuple)) and len(location) == 2:
        lat, lon = location
    else:
        raise TypeError("location must be a dict with 'lat' and 'lon' or a list/tuple [lat, lon]")

    lat_str = f"{abs(lat)}°{'N' if lat >= 0 else 'S'}"
    lon_str = f"{abs(lon)}°{'E' if lon >= 0 else 'W'}"
    return f"{lat_str}, {lon_str}"




def add_lat_markers(ax, fontscale: float = 1, side: str = "left"):
    """
    Adds latitude markers to the specified side of the plot.
    
    Parameters:
    ax : matplotlib.axes.Axes
        The axis to modify.
    fontscale : float, optional
        Scale factor for the font size of labels (default is 1).
    side : str, optional
        Side to place labels on ('left' or 'right', default is 'left').
    """
    special_lats = [23.5, -23.5, 67.5, -67.5]
    special_lats_str = [str(lat) + r'$^\circ$' + ('N' if lat >= 0 else 'S') for lat in special_lats]
    
    ax.set_yticks(special_lats)
    ax.set_yticklabels(special_lats_str, fontsize=8 * fontscale)
    
    if side == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    else:
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
    
    ax.set_ylabel('')

def add_lon_markers(ax, fontscale: float = 1):
    """
    Adds longitude markers to the plot.
    
    Parameters:
    ax : matplotlib.axes.Axes
        The axis to modify.
    fontscale : float, optional
        Scale factor for the font size of labels (default is 1).
    """
    special_lon = np.arange(-180, 210, 30)
    special_lon_str = [str(lon) + r'$^\circ$' + ('E' if lon >= 0 else 'W') for lon in special_lon]
    
    ax.set_xticks(special_lon)
    ax.set_xticklabels(special_lon_str, fontsize=8 * fontscale)
    ax.set_xlabel('')

def add_lat_lon_markers(ax, fontscale: float = 1, side: str = "left"):
    """
    Adds both latitude and longitude markers to the plot.
    
    Parameters:
    ax : matplotlib.axes.Axes
        The axis to modify.
    fontscale : float, optional
        Scale factor for the font size of labels (default is 1).
    side : str, optional
        Side to place latitude labels on ('left' or 'right', default is 'left').
    """
    add_lat_markers(ax, fontscale, side)
    add_lon_markers(ax, fontscale)


def create_levels(vmax:float, vmin:float=None, step:float=1)->np.ndarray:
    '''
    Ensures that all instances of creating levels using vmax + step as the max.
    '''
    vmin = -vmax if vmin is None else vmin
    return np.arange(vmin, vmax + step, step)

def add_figure_label(ax: plt.Axes, label:str, font_scale:int=1, x:float=0.01, y:float=1.05):
    ax.annotate(label, xy = (x,y), xycoords = 'axes fraction', size=PlotConfig.label_size*font_scale)

def format_axis(ax: plt.Axes, title:str=None, xlabel:str=None, ylabel:str=None, invisible_spines=None, 
               font_scale=1, rotation=0, labelpad=100, xlabelpad=10, grid:bool=True):
    '''Formatting with no top and right axis spines and correct tick size.'''
    if xlabel: ax.set_xlabel(xlabel, fontsize=PlotConfig.label_size*font_scale, ha='center', va='center',
                            labelpad=xlabelpad)
    if ylabel: ax.set_ylabel(ylabel, rotation=rotation, labelpad=labelpad*font_scale,
                             fontsize=PlotConfig.label_size*font_scale, ha='center', va='center')
    if title: ax.set_title(title, fontsize=PlotConfig.title_size*font_scale)
    ax.tick_params(axis='x', labelsize=PlotConfig.tick_size*font_scale)
    ax.tick_params(axis='y', labelsize=PlotConfig.tick_size*font_scale)
    if invisible_spines: [ax.spines[spine].set_visible(False) for spine in invisible_spines]
    if grid: ax.grid(True, alpha=0.5, c='grey', linestyle='--')
    return ax


def format_lat_lon(ax):
    """
    Add latitude and longitude gridlines and labels to a Cartopy axes.

    Parameters:
        ax (cartopy.mpl.geoaxes.GeoAxesSubplot): The Cartopy axes to format.

    Returns:
        None
    """
    # Add latitudes and longitudes
    gl = ax.gridlines(draw_labels=True, linestyle='--')

    # Do not show labels on top and right sides
    gl.xlabels_top = False
    gl.ylabels_right = False

    # Set the formatter for longitude and latitude labels
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

def match_ticks(ax1, ax2, perturb=0, ax2_round_level=1, logginglevel='ERROR'):
    """
    Match the tick values and limits of two matplotlib axes.

    This function is used to synchronize the y-axis ticks and limits of two
    subplots in a Matplotlib figure.

    Parameters:
    ax1 (matplotlib.axes._subplots.AxesSubplot): The first subplot.
    ax2 (matplotlib.axes._subplots.AxesSubplot): The second subplot.


    Returns:
    None
    """
    utils.change_logginglevel(logginglevel)
    
    # Get the tick values of both axes
    ax1_yticks = ax1.get_yticks()
    ax2_yticks = ax2.get_yticks()
    logger.debug('Current ticks')
    logger.debug(ax1_yticks)
    logger.debug(ax2_yticks)

    # perturb = perturb * 2\
    perturbation_factor = perturb * (ax2_yticks[-1] - ax2_yticks[0])
    logger.debug(f'{perturbation_factor=}')
    # Set the y-axis limits of both axes to the first and last ticks
    new_ax2_ylims = np.take(ax2_yticks, [1, -2]) - np.array([-perturbation_factor if ax2_yticks[0] < 0 else perturbation_factor, perturbation_factor])

    
    ax2.set_ylim(new_ax2_ylims)
    logger.debug('New ticks')
    logger.debug(new_ax2_ylims)

    new_ax2_yticks = np.linspace(*new_ax2_ylims, len(ax1_yticks)-2)
    logger.debug(new_ax2_yticks)
    
    ax2.set_yticks(new_ax2_yticks)
    logger.debug(new_ax2_yticks)
    
    ax2.set_ylim(new_ax2_ylims)
    logger.debug(new_ax2_ylims)
    ax2.set_yticklabels(np.round(new_ax2_yticks, ax2_round_level))


    return ax1, ax2

def clip_axis_ends(ax):
    """
    Clip the first and last y-axis tick labels on a Matplotlib Axes object.

    Parameters:
    ax (matplotlib.axes._subplots.Axes): The Axes object for which you want to modify the y-axis tick labels.

    Returns:
    None

    This function retrieves the current y-axis tick labels from the provided Axes object and removes the text from
    the first and last tick labels, effectively clipping the ends of the y-axis.

    Example:
    import matplotlib.pyplot as plt

    # Create a simple plot
    plt.plot([1, 2, 3, 4, 5], [10, 20, 25, 30, 35])

    # Get the current Axes object
    ax = plt.gca()

    # Call clip_axis_ends to clip the y-axis tick labels
    clip_axis_ends(ax)

    # Display the modified plot
    plt.show()
    """
    # Get the current y-axis tick labels
    labels = [label.get_text() for label in ax.get_yticklabels()]

    # Clip the first and last y-axis tick labels by setting them to empty strings
    labels[0] = ''
    labels[-1] = ''

    # Set the modified y-axis tick labels back to the Axes object
    ax.set_yticklabels(labels)

def fig_formatter(height_ratios: List[float] , width_ratios: List[float],  hspace:float = 0.4, wspace:float = 0.2):
    
    height = np.sum(height_ratios)
    width = np.sum(width_ratios)
    num_rows = len(height_ratios)
    num_cols = len(width_ratios)
    
    fig  = plt.figure(figsize = (10*width, 5*height)) 
    gs = gridspec.GridSpec(num_rows ,num_cols, hspace=hspace, 
                           wspace=wspace, height_ratios=height_ratios, width_ratios=width_ratios)
    return fig, gs



def create_discrete_cmap(cmap, number_divisions:int=None, levels=None, vmax=None, vmin=None, step=1,
                         add_white:bool=False, white_loc='start', clip_ends:int=0):
    '''
    Creates a discrete color map of cmap with number_divisions
    '''
    
    if levels is not None: number_divisions = len(levels)
    elif vmax is not None: number_divisions = len(create_levels(vmax, vmin, step))
                
    color_array = plt.cm.get_cmap(cmap, number_divisions+clip_ends)(np.arange(number_divisions+clip_ends)) 

    if add_white:
        if white_loc == 'start':
            white = [1,1,1,1]
            color_array[0] = white
        elif white_loc == 'middle':
            upper_mid = np.ceil(len(color_array)/2)
            lower_mid = np.floor(len(color_array)/2)

            white = [1,1,1,1]

            color_array[int(upper_mid)] = white
            color_array[int(lower_mid)] = white

            # This must also be set to white. Not quite sure of the reasoning behind this. 
            color_array[int(lower_mid) - 1] = white
        
    cmap = mpl.colors.ListedColormap(color_array)
    
    return cmap


def create_colorbar(plot, cax, levels, tick_offset=None, cut_ticks=1, round_level=2,
                    font_scale=1, cbar_title='', orientation='horizontal', logginglevel='ERROR', **kwargs):
    """
    Create and customize a colorbar for a given plot.

    Parameters:
        plot: matplotlib plot object
            The plot that the colorbar is associated with.
        cax: matplotlib axes object
            The colorbar axes.
        levels: array-like
            The levels used in the plot.
        tick_offset: str, optional
            Offset method for ticks ('center' or None). Default is None.
        cut_ticks: int, optional
            Frequency of ticks to cut for better visualization. Default is 1.
        round_level: int, optional
            Number of decimal places to round tick labels to. Default is 2.
        font_scale: float, optional
            Scaling factor for font size. Default is 1.
        cbar_title: str, optional
            Title for the colorbar. Default is an empty string.
        orientation: str, optional
            Orientation of the colorbar ('horizontal' or 'vertical'). Default is 'horizontal'.
        **kwargs:
            Additional keyword arguments for colorbar customization.

    Returns:
        cbar: matplotlib colorbar object
            The customized colorbar.
    """
    
    utils.change_logginglevel(logginglevel)
    logger.info(utils.function_name())
    logger.info(f'**{__file__}')
    logger.debug(locals())
    # Create the colorbar with specified orientation and other keyword arguments
    cbar = plt.colorbar(plot, cax=cax, orientation=orientation, **kwargs) # , ticks=levels
    # cbar.ax.tick_params(axis='both', which='both', labelsize=tick_size)
    
    # Calculate tick locations and labels based on tick_offset and cut_ticks settings
    tick_locations = levels
    tick_labels = levels
    logger.debug(f'{tick_labels=}\n{tick_locations=}')
    if tick_offset == 'center':
        tick_locations = levels[:-1] + np.diff(levels) / 2
        tick_labels = tick_labels[:-1]
    
    if cut_ticks > 1:
        tick_locations = tick_locations[::cut_ticks]
        tick_labels = tick_labels[::cut_ticks]
    
    logger.debug(f'{tick_labels=}\n{tick_locations=}')
    
    # Set tick locations and labels on the colorbar
    cbar.set_ticks(tick_locations)
    tick_labels = np.round(tick_labels, round_level)
    
    logger.info(f'{tick_labels=}')
    
    # Customize colorbar based on orientation
    if orientation == 'horizontal':
        cbar.ax.xaxis.set_ticks(tick_locations, minor=True)
        cbar.ax.set_xticklabels(tick_labels, fontsize=14*font_scale)
        cbar.ax.set_title(cbar_title, size=18 * font_scale)

    else:
        cbar.ax.set_yticks(tick_locations)
        cbar.ax.set_yticklabels(tick_labels, fontsize=10*font_scale, rotation=90)
        cbar.ax.set_ylabel(cbar_title, size=12 * font_scale, rotation=0, labelpad=30)
        
    return cbar


def style_plot(ax, legend_loc='upper left', legend_bbox=None, **kwargs):
    # Set defaults with flexible kwargs
    grid_linewidth = kwargs.get('grid_linewidth', 0.5)
    grid_linestyle = kwargs.get('grid_linestyle', '--')
    grid_alpha = kwargs.get('grid_alpha', 0.7)
    facecolor = kwargs.get('facecolor', '#f9f9f9')
    fontsize = kwargs.get('fontsize', 12)
    xtick_spacing = kwargs.get('xtick_spacing', 5)
    hide_alternate_xticks = kwargs.get('hide_alternate_xticks', True)
    
    # Apply grid and background style
    ax.grid(True, which='both', linestyle=grid_linestyle, linewidth=grid_linewidth, alpha=grid_alpha)
    ax.set_facecolor(facecolor)
    
    # Customize axis labels
    ax.set_ylabel(kwargs.get('ylabel', ''), fontsize=fontsize)
    ax.set_xlabel(kwargs.get('xlabel', ''), fontsize=fontsize)
    
    # Customize legend
    if ax.get_legend_handles_labels()[0]:  # Only add if legend items exist
        if legend_bbox:
            ax.legend(loc=legend_loc, fontsize=kwargs.get('legend_fontsize', 10), frameon=False, bbox_to_anchor=legend_bbox)
        else:
            ax.legend(loc=legend_loc, fontsize=kwargs.get('legend_fontsize', 10), frameon=False)
    
    # Set and style x-ticks (assuming time axis for consistency)
    time_values = ax.get_lines()[0].get_xdata()  # Automatically detect x-axis data
    xticks = np.arange(*np.take(time_values, [0, -1]), xtick_spacing)
    ax.set_xticks(xticks)
    
    if hide_alternate_xticks:
        xtick_labels = xticks.astype(str)
        xtick_labels[::2] = ''  # Hide every other label for cleaner look
        ax.set_xticklabels(xtick_labels)



def format_colobrar(cbar, title='', fontscale=1, pad=20):
    cbar.ax.set_title(title, fontsize=8 * fontscale, pad=pad)
    
    # Scale y-tick labels' font size
    cbar.ax.tick_params(axis='x', labelsize=8 * fontscale)


def replace_slice(arr: np.ndarray, keep: slice = slice(None)) -> np.ndarray:
    """Replace all elements with '' except those in the given slice."""
    arr_to_return = np.full_like(arr, '', dtype=object) 
    arr_to_return[keep] = arr[keep] 
    return arr_to_return


def create_discrete_colorbar(
    cmap, levels, cax, label, orientation='vertical', fontscale=1,
    pad=20, tick_slice=slice(None), tick_round=2, **kwargs):
    """
    Create a discrete colorbar with major and minor ticks.

    Args:
        cmap: Colormap to use.
        levels: Discrete levels (boundaries).
        cax: Colorbar axis.
        label: Label for the colorbar.
    """
    norm = mpl.colors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Create colorbar
    cbar = plt.colorbar(sm, cax=cax, orientation=orientation, **kwargs)

    # Round and apply tick slicing
    levels = levels.round(tick_round)
    major_ticks = levels[tick_slice]  # Major ticks
    minor_ticks = levels  # All levels as minor ticks
    # ticklabels = replace_slice(levels, tick_slice)
    ticklabels = major_ticks
    # Ensure equal length
    # if len(major_ticks) != len(ticklabels):
    #     ticklabels = ticklabels[:len(major_ticks)]

    cbar.set_ticks(major_ticks)
    cbar.set_ticklabels(ticklabels)

    cbar.ax.set_title(label, fontsize=12 * fontscale, pad=pad)

    # Add minor ticks (without labels)
    cbar.ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(minor_ticks))

    # Scale y-tick labels' font size
    if orientation == 'verital':
        cbar.ax.tick_params(axis='y', labelsize=10 * fontscale)
    else:
        cbar.ax.tick_params(axis='x', labelsize=10 * fontscale)


    return cbar

# def create_discrete_colorbar(
#     cmap, levels, cax, label, orientation='vertical', fontscale=1,
#     pad = 20, tick_slice=slice(None), tick_round = 2,
#     **kwargs):
#     """
#     Create a discrete colorbar with the given colormap and levels.
    
#     Args:
#         cmap: Colormap to use.
#         levels: Discrete levels (boundaries).
#         cax: Colorbar axis.
#         label: Label for the colorbar.
#     """
#     norm = mpl.colors.BoundaryNorm(boundaries=levels, ncolors=cmap.N)
#     sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    
#     # Use plt.colorbar when cax is provided, no need for fig
#     cbar = plt.colorbar(sm, cax=cax, orientation=orientation, **kwargs)
    
#     cbar.set_ticks(levels)
#     # ticklabels = levels.astype(str)
#     # ticklabels[1::2] = ''
#     levels = levels.round(tick_round)
#     ticklabels = replace_slice(levels, tick_slice)
#     cbar.set_ticklabels(ticklabels)
#     cbar.ax.set_title(label, fontsize=12 * fontscale, pad=pad)
    
#     # Scale y-tick labels' font size
#     cbar.ax.tick_params(axis='y', labelsize=10 * fontscale)
    
#     return cbar


def hatch(ax, ds, **kwargs):
    invert = lambda ds: xr.where(ds, 0, 1)

    ds = invert(ds)
    LON, LAT = np.meshgrid(ds.lon.values, ds.lat.values)
    ax.contourf(LON, LAT, ds.values, levels=[-1, 0, 1, 2], **kwargs)

