import numpy as np
import xarray as xr

import matplotlib.pyplot as plt

import os, sys
# My imports
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts'))
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'time_of_emergene_drafts', 'src'))

import toe_calc



def generate_rel_freq_between_year(da, start, end, bins=None):
    arr = (da
          .where(
              (da.time.dt.year > start) & (da.time.dt.year < end),
              drop=True)
          .values)

    bins, rel_freq = toe_calc.discrete_pdf(arr, bins)

    return bins, rel_freq


def plot_bar(bins, rel_freq, ax=None, **kwargs):
    # Identify indices where rel_freq is not zero
    non_zero_indices = np.where(rel_freq != 0)[0]

    # Subset the bins and rel_freq to exclude 0-padding
    start, end = non_zero_indices[0], non_zero_indices[-1] + 1  # Include the rightmost bin
    bins_subset = bins[start:end + 1]  # +1 to include the right edge bin
    rel_freq_subset = rel_freq[start:end]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.bar(bins_subset[:-1], rel_freq_subset, width=np.diff(bins_subset), align='edge', alpha=0.5, edgecolor='k')

    return ax
    
# def plot_bar(bins, rel_freq , ax=None, **kwargs):

#     if ax == None:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)

#     ax.bar(bins[:-1], rel_freq, width=np.diff(bins), align='edge', alpha=0.5, edgecolor='k') #, label=kwargs.get('label', None)