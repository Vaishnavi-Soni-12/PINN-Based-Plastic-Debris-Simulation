import xarray as xr
import numpy as np
import json
import os

CHIRPS_PATH = os.path.abspath(
    "../../data/chirps/chirps-v2.0.2022.monthly.nc"
)

ds = xr.open_dataset(CHIRPS_PATH)
rain = ds["precip"]

mean_rain = float(rain.mean())
p25 = float(rain.quantile(0.25))
p75 = float(rain.quantile(0.75))

scaling = {
    "dry": round(p25 / mean_rain, 2),
    "normal": 1.0,
    "wet": round(p75 / mean_rain, 2)
}

os.makedirs("../outputs", exist_ok=True)
with open("../outputs/climate_scaling.json", "w") as f:
    json.dump(scaling, f, indent=2)

print("✔ Climate scaling derived from CHIRPS")
print(scaling)
