from enum import Enum
from typing import NamedTuple
from abc import ABC
from dataclasses import dataclass, fields
from typing import Optional


@dataclass(frozen=True)
class BasePeriod(ABC):
    start: int
    end: int
    @property
    def length(self) -> int:
        return self.end - self.start
    @property
    def value(self) -> tuple:
        return (self.start, self.end)


class YearRange(Enum):
    MODERN_PERIOD = (1959, 1989)
    MID_20TH_CENTURY = (1929, 1959)
    EARLY_20TH_CENTURY = (1899, 1929)
    ERA5_START = (1940, 1970)
    

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


@dataclass(frozen=True)
class ThresholdProfileBase(ABC):
    sn_threshold: Optional[int] = None
    pvalue_threshold: Optional[float] = None
    overlap_threshold: Optional[int] = None
    hd_threshold: Optional[int] = None

    def __repr__(self) -> str:
        field_values = ", ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))
        return f"{self.__class__.__name__}({field_values})"

@dataclass(frozen=True)
class ThresholdProfileUnusual(ThresholdProfileBase):
    sn_threshold: int = 1
    pvalue_threshold: float = 0.01
    overlap_threshold: int = 62
    hd_threshold: int = 33

@dataclass(frozen=True)
class ThresholdProfileUnfamiliar(ThresholdProfileBase):
    sn_threshold: int = 2
    overlap_threshold: int = 32
    hd_threshold: int = 66

@dataclass(frozen=True)
class ThresholdProfileUnknown(ThresholdProfileBase):
    sn_threshold: int = 3
    overlap_threshold: int = 13
    hd_threshold: int = 82


PVALUE_THESHOLD1 = 0.01
OVERLAP_THRESHOLD = 62
HD_THRESHOLD = 33
SN_THRESHOLD1 = 1

PVALUE_TESTS = ['ks', 'ttest', 'anderson']
OVERLAP_TESTS = ['frac', 'perkins']
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
    'era5_cape': 'ERA5 CAPE',
    'access_pr': 'ACCESS Precip (SSP5-8.5)',
    'access_ssp585_r10i1p1f1_pr_QSDEC': 'ACCESS Precip\n(Boreal Winter, SSP5-8.5)',
    'access_ssp585_r10i1p1f1_pr_QSJUN': 'ACCESS Precip\n(Austral Winter, SSP5-8.5)'
}


NAME_CONVERSION_DICT_SHORT = {
    'sn': 'S/N',#'Signal-to-Noise Ratio'
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


kde_kwargs= dict(bw_method=0.2) # silverman, scott#bw_method=0.2)

class LocationBoxes(Enum):
    NORTH_AMERICA = dict(lat=slice(10, 66), lon=slice(-160, -49))
    NORTH_AMERICA_LAND_TARGET = dict(lat=slice(10, 66), lon=slice(-120, -80))
    ASIA = dict(lat=slice(-15, 40), lon=slice(60, 140))


class Region(NamedTuple):
    name: str
    latlon: tuple

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