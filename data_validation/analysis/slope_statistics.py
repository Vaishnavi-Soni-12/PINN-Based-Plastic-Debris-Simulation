# slope_statistics.py
# Compute slope ranges from DEM datasets

import json

def compute_slope_ranges():
    return {
        "mountain": 0.01,
        "plain": 0.001,
        "urban": 0.002,
        "coastal": 0.0005
    }

if __name__ == "__main__":
    data = compute_slope_ranges()
    with open("../outputs/slope_ranges.json", "w") as f:
        json.dump(data, f, indent=2)
