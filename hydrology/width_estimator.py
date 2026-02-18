# hydrology/width_estimator.py

WIDTH_RANGES = {
    "mountain": (10, 50),
    "plain": (50, 500),
    "urban": (20, 100),
    "coastal": (200, 800)
}

def estimate_width(river_type):
    low, high = WIDTH_RANGES[river_type]
    return (low + high) / 2
