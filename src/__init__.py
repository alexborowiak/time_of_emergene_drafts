# src/__init__.py

from deprecated import time_of_emergence_calc, time_of_emergence_data_analysis  # Re-expose at top-level

# Optional: Explicitly define what gets imported when using `from src import *`
__all__ = [
    "climate_utils",
    "paths",
    "plotting_utils",
    "toe_calc",
    "toe_constants",
    "toe_data_analysis",
    "toe_plots",
    "utils",
]

# Optional: Add a deprecation warning for `deprecated` modules
import warnings
warnings.warn(
    "Some modules have been moved to the 'deprecated' folder and may be removed in future versions.",
    category=DeprecationWarning,
    stacklevel=2,
)
