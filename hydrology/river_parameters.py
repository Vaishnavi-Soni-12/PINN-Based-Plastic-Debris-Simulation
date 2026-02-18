# hydrology/river_parameters.py
from core.config_loader import RIVER_WIDTH_CLASSES, CLIMATE_SCALING

def get_width_range(river_type):
    return tuple(RIVER_WIDTH_CLASSES[river_type])

def get_climate_multiplier(season):
    return CLIMATE_SCALING[season]
