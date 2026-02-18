# data

## Purpose

Stores static runtime constants and local raw reference datasets used by validation and model parameterization.

## Subfolders

- `static/`
  - Versioned JSON constants used directly by runtime modules.
- `cache/`
  - Local temporary cache space (not versioned).
- `chirps/`
  - Local CHIRPS NetCDF rainfall dataset files.
- `grwl/`
  - Local GRWL shapefile components.
- `hydrosheds/`
  - Local HydroSHEDS raster datasets and docs.

## Runtime-Critical Files

- `static/river_width_classes.json`
- `static/climate_scaling.json`
- `static/manning_coefficients.json` (reserved for hydrology extensions)

## Version Control Policy

- Commit `data/static/*.json`.
- Do not commit large raw data in `chirps/`, `grwl/`, and `hydrosheds/`.
