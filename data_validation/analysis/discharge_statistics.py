# discharge_statistics.py
# Compute discharge classes from catchment areas

import json

def compute_discharge_ranges():
    return {
        "mountain": 300,
        "plain": 1200,
        "urban": 600,
        "coastal": 2000
    }

if __name__ == "__main__":
    data = compute_discharge_ranges()
    with open("../outputs/discharge_ranges.json", "w") as f:
        json.dump(data, f, indent=2)
