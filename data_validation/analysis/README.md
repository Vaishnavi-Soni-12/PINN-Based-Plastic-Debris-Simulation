# data_validation/analysis

## Purpose

Contains offline analysis scripts that convert raw datasets into compact, reusable constants.

## Scripts

- `width_statistics.py`
  - Reads GRWL shapefile and derives terrain-based width ranges.
  - Writes `../outputs/width_ranges.json`.
- `chirps_validation.py`
  - Reads CHIRPS precipitation data and computes dry/normal/wet scaling factors.
  - Writes `../outputs/climate_scaling.json`.
- `slope_statistics.py`
  - Placeholder slope class generator.
  - Writes `../outputs/slope_ranges.json`.
- `discharge_statistics.py`
  - Placeholder discharge class generator.
  - Writes `../outputs/discharge_ranges.json`.

## Execution

Run scripts from this directory so relative output paths resolve correctly:

```powershell
python width_statistics.py
python chirps_validation.py
python slope_statistics.py
python discharge_statistics.py
```

## Output Handling

Review generated JSON values before manually updating `data/static/`.
