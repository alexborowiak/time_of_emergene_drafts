# Time of Emergence Drafts

This repository contains draft notebooks, modules, and scripts for investigating Time of Emergence (ToE) in climate datasets. The work contributes to a technical review on Time of Emergence, prepared for submission to *Nature Reviews Earth & Environment*.

ToE marks the point at which a climate signal becomes distinguishable from internal variability, providing a framework for evaluating the detectability of anthropogenic change. This project assesses a range of ToE methods across synthetic data, observational records, and large ensemble climate models.

## Directory Structure

```
time_of_emergence_drafts/
├── draft_notebooks/         # Early-stage exploratory notebooks
├── src/                     # Core analysis modules
├── toplevel_notebooks/      # Primary notebooks for figures and methods
├── constants.py             # Shared constants
├── README.md                # Project overview
```

Additional contents include:

- `toplevel_notebooks/`  
  Contains main analysis notebooks:
  - `00_data_processing.ipynb`  
  - `01_global_mean_emergence_V2.ipynb`  
  - `03_calculations_ToE_main.ipynb`  
  - `03_calculations_ToE_other_base_period_resolution_aggregation.ipynb`  
  - `05_toe_plots_v4.ipynb`  
  - `relationship_SN_KS_TTest.ipynb`  
  - and others.

- `src/`  
  Active codebase:
  - `toe_calc.py`, `toe_data_analysis.py`, `toe_plots.py`  
  - Supporting utilities: `paths.py`, `open_data.py`, `plotting_utils.py`, etc.  
  - `deprecated/` subfolder contains legacy or experimental scripts (e.g. `toe_calc_variations.py`, `toe_constants.py`).

## Methods Implemented

- Signal-to-noise ratio (S/N), including LOWESS and rolling approaches  
- Kolmogorov–Smirnov (KS) test  
- Parametric t-test  
- Area of Overlap (AO)  
- Hellinger Distanc
