import geopandas as gpd
import numpy as np
import json
import os

GRWL_PATH = os.path.abspath(
    "../../data/grwl/GRWL_summaryStats.shp"
)

gdf = gpd.read_file(GRWL_PATH)

# Extract width statistics (meters)
w_min = float(np.nanmedian(gdf["width_min_"]))
w_med = float(np.nanmedian(gdf["width_med_"]))
w_mean = float(np.nanmedian(gdf["width_mean"]))
w_max = float(np.nanmedian(gdf["width_max_"]))

print("Derived GRWL statistics (meters):")
print("Min:", w_min, "Median:", w_med, "Mean:", w_mean, "Max:", w_max)

# Regime-based abstraction (documented)
width_ranges = {
    "mountain": [round(w_min, 1), round(w_med * 0.7, 1)],
    "plain": [round(w_med * 0.7, 1), round(w_med * 1.5, 1)],
    "urban": [round(w_med * 0.5, 1), round(w_med, 1)],
    "coastal": [round(w_med * 1.5, 1), round(w_max, 1)]
}

os.makedirs("../outputs", exist_ok=True)
with open("../outputs/width_ranges.json", "w") as f:
    json.dump(width_ranges, f, indent=2)

print("✔ River width ranges derived from GRWL")
print(width_ranges)
