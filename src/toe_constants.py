from enum import Enum
from typing import NamedTuple


class YearRange(Enum):
    MODERN_PERIOD = (1959, 1989)
    MID_CENTURY = (1900, 1930)

    @property
    def start(self):
        return self.value[0]

    @property
    def end(self):
        return self.value[1]
    @property
    def length(self):
        return self.end - self.start


# The threshold of emergence for the different tests
PVALUE_THESHOLD1 = 0.01
OVERLAP_THRESHOLD = 62
SN_THRESHOLD1 = 1

PVALUE_TESTS = ['ks', 'ttest', 'anderson']
OVERLAP_TESTS = ['frac', 'perkins', 'hd']
SN_TYPES = ['sn', 'sn_lowess', 'sn_poly', 'sn_average', 'nn']


EMERGENCE_THRESHOLD_DICT = {
    'sn_lowess': SN_THRESHOLD1,
    'sn': SN_THRESHOLD1,
    'ks': PVALUE_THESHOLD1,
    'anderson': PVALUE_THESHOLD1,
    'ttest': PVALUE_THESHOLD1,
    'frac': OVERLAP_THRESHOLD,
    'perkins': OVERLAP_THRESHOLD
}

NAME_CONVERSION_DICT = {
    'sn': 'S/N Ratio (LOWESS)',#'Signal-to-Noise Ratio'
    'sn_lowess_base': 'S/N Ratio (LOWESS, Base Noise)',
    'sn_lowess_full': 'S/N Ratio (LOWESS, Full Series Noise)',
    'sn_lowess': 'S/N Ratio (LOWESS)',#'Signal-to-Noise Ratio'
    'sn_lowess_rolling': 'S/N Ratio (LOWESS, Rolling)',#'Signal-to-Noise Ratio'
    'sn_rolling': 'S/N Ratio (Rolling Mean)',#'Signal-to-Noise Ratio'
    'sn_anom': 'S/N Ratio (Anomalies)',#'Signal-to-Noise Ratio'
    'sn_poly4': 'S/N Ratio (4th Order Polynomial)',#'Signal-to-Noise Ratio'
    'sn_lowess_rolling_smooth': 'S/N Ratio (LOWESS, Rolling, smooth)',#'Signal-to-Noise Ratio'
    'nn': 'New Normal',
    'ks': 'Kolmogorov-Smirnov',
    'ttest': 'T-Test',
    'anderson': 'Anderson-Darling',
    'perkins': 'Perkins Skill Score', 
    'frac': 'Fractional Geometric Area',
    'hd': 'Hellinger Distance'    
}

VARIABLE_CONVERSION_DICT = {
    'best_temperature': 'BEST Temp',
    'era5_t2m': 'ERA5 2m Temp',
    'access_pr': 'ACCESS Precip (SSP5-8.5)'
}


NAME_CONVERSION_DICT_SHORT = {
    'sn': 'S/N (LOWESS)',#'Signal-to-Noise Ratio'
    'sn_lowess': 'S/N (LOWESS)',#'Signal-to-Noise Ratio'
    'sn_rolling': 'S/N (Mean)',#'Signal-to-Noise Ratio'
    'sn_anom': 'S/N (Anom)',#'Signal-to-Noise Ratio'
    'sn_poly4': 'S/N (4th-Poly)',#'Signal-to-Noise Ratio'
    'nn': 'NN',
    'ks': 'KS', #Kolmogorov-Smirnov
    'ttest': 'T-Test',
    'anderson': 'Anderson-Darling',
    'perkins': 'PSS', 
    'frac': 'FGA',
    'hd': 'HD'    
}


class LocationBoxes(Enum):
    NORTH_AMERICA = dict(lat=slice(10, 66), lon=slice(-160, -49))
    NORTH_AMERICA_LAND_TARGET = dict(lat=slice(10, 66), lon=slice(-120, -80))
    ASIA = dict(lat=slice(-15, 40), lon=slice(60, 140))


class Region(NamedTuple):
    name: str
    slice: tuple

class regionLatLonTuples(Enum):
    GLOBAL = Region('global', slice(None, None))
    LAND = Region('land', slice(None, None))
    OCEAN = Region('ocean', slice(None, None))
    NH = Region('nh', slice(0, None))
    SH = Region('sh', slice(None, 0))
    TROPICS = Region('tropics', slice(-23, 23))
    MID_LAT_SH = Region('mid_lat_sh', slice(-66, -23))
    MID_LAT_NH = Region('mid_lat_nh', slice(23, 66))
    ARCTIC = Region('arctic', slice(66, None))
    ANTARCTIC = Region('antarctic', slice(None, -66))


NAMING_MAP = {
    'global': 'Global',
    'nh': 'Northern Hemisphere',
    'sh': 'Southern Hemisphere',
    'tropics': 'Tropics',
    'land': 'Land',
    'ocean': 'Ocean',
    'mid_lat_nh': 'Mid Latitudes NH',
    'mid_lat_sh': 'Mid Latitudes SH',
    'arctic': 'Arctic',
    'antarctic': 'Antarctic'}