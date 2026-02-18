# hydrology/discharge_estimator.py

DISCHARGE_BASE = {
    "mountain": 300,
    "plain": 1200,
    "urban": 600,
    "coastal": 2000
}

RAIN_SCALE = {
    "dry": 0.7,
    "normal": 1.0,
    "wet": 1.3
}

def estimate_discharge(river_type, season):
    return DISCHARGE_BASE[river_type] * RAIN_SCALE[season]
