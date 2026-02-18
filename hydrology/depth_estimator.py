# hydrology/depth_estimator.py

MANNING_N = {
    "mountain": 0.035,
    "plain": 0.03,
    "urban": 0.025,
    "coastal": 0.04
}

SLOPE = {
    "mountain": 0.01,
    "plain": 0.001,
    "urban": 0.002,
    "coastal": 0.0005
}

def estimate_depth(width, velocity, river_type):
    n = MANNING_N[river_type]
    s = SLOPE[river_type]
    return (velocity * n / (s ** 0.5)) ** (3/2)
