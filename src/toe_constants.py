from enum import Enum

VALUE_TESTS = ['ks', 'ttest', 'anderson']


NAME_CONVERSION_DICT = {'signal_to_noise': 'Signal to Noise Ratio',
 'ks': 'Kolmogorov-Smirnov',
 'ttest': 'T-Test',
 'anderson': 'Anderson-Darling'}


class LocationBoxes(Enum):
    NORTH_AMERICA = dict(lat=slice(10, 66), lon=slice(-160, -49))
    NORTH_AMERICA_LAND_TARGET = dict(lat=slice(10, 66), lon=slice(-120, -80))
    ASIA = dict(lat=slice(-15, 40), lon=slice(60, 140))