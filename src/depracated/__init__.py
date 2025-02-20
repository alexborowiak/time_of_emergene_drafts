# src/deprecated/__init__.py

import warnings

warnings.warn(
    "Modules in 'deprecated' are deprecated and may be removed in future versions.",
    category=DeprecationWarning,
    stacklevel=2,
)

# Optional: Define what can be imported from `deprecated`
__all__ = [
    "time_of_emergence_calc",
    "time_of_emergence_data_analysis",  # Include any other deprecated modules
]
