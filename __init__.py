import warnings

# Backward compatibility for renamed modules
warnings.warn(
    "Modules like 'time_of_emergence_plots' have been renamed (e.g., to 'toe_plots'). "
    "Please update your imports to avoid deprecation issues.",
    DeprecationWarning
)

# Old to new aliases for backward compatibility
from .toe_plots import * as time_of_emergence_plots
from .toe_constants import * as toe_constants

# If other files were renamed, add similar aliases:
# from .new_module_name import * as old_module_name

# Optional: Import key functions or classes to make them available when importing the package
__all__ = [
    # Expose specific functions, classes, or modules to simplify imports
    'time_of_emergence_plots',
    'toe_constants',
    'climate_utils',       # Expose climate-related utilities
    'diagnostic_tools',    # Expose diagnostic tools
    'plotting_utils',      # Expose general plotting functions
    'paths',               # For any path-related configuration
    'my_stats',            # For statistical utilities
    'misc',                # For miscellaneous utilities
    'open_data'            # Any data access utilities
]
