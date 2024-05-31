from enum import Enum
from typing import NamedTuple

# The threshold of emergence for the different tests
PVALUE_THESHOLD1 = 0.01
OVERLAP_THRESHOLD = 62
SN_THRESHOLD1 = 1

PVALUE_TESTS = ['ks', 'ttest', 'anderson']
OVERLAP_TESTS = ['frac', 'perkins']


EMERGENCE_THRESHOLD_DICT = {
    'signal_to_noise': SN_THRESHOLD1,
    'ks': PVALUE_THESHOLD1,
    'anderson': PVALUE_THESHOLD1,
    'ttest': PVALUE_THESHOLD1,
    'frac': OVERLAP_THRESHOLD,
    'perkins': OVERLAP_THRESHOLD
}

NAME_CONVERSION_DICT = {
    'signal_to_noise': 'S/N Ratio',#'Signal-to-Noise Ratio'
    'ks': 'KS', #Kolmogorov-Smirnov
    'ttest': 'T-Test',
    'anderson': 'Anderson-Darling',
    'perkins': 'Perkins Skill Score', 
    'frac': 'Fractional Geometric Area'}


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