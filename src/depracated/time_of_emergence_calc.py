import warnings
from toe_calc import *  # Import everything from the new module

warnings.warn(
    "'time_of_emergence_calc' is deprecated and will be removed in future versions. "
    "Use 'toe_calc' instead.",
    category=DeprecationWarning,
    stacklevel=2,
)
