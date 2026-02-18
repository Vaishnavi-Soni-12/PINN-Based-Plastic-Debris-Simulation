import numpy as np
from hydrology.river_parameters import get_width_range

def latlon_to_local_xy(lat, lon, length_m=1000.0):
    x = (lon % 1) * length_m
    y = (lat % 1) * (length_m / 4)
    return x, y

def create_spatial_domain(
    river_type,
    lat,
    lon,
    nx=200
):
    w_min, w_max = get_width_range(river_type)
    width = 0.5 * (w_min + w_max)

    x0, y0 = latlon_to_local_xy(lat, lon)

    x = np.linspace(x0, x0 + 1000, nx)
    y = np.linspace(y0, y0 + width, int(nx * width / 1000))

    return x, y
