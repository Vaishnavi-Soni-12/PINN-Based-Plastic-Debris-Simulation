# hydrology

## Purpose

Encapsulates hydrology-related constants and estimators for river width, discharge, depth, velocity, diffusion, and climate scaling.

## Files

- `river_parameters.py`
  - Primary interface used by runtime modules.
  - Reads width and climate constants from `data/static/` via `core/config_loader.py`.
- `width_estimator.py`
  - Rule-based width estimation by terrain class.
- `discharge_estimator.py`
  - Terrain baseline discharge scaled by seasonal regime.
- `depth_estimator.py`
  - Manning-style depth estimation from width, velocity, and slope assumptions.
- `velocity_model.py`
  - Velocity from discharge, width, and depth.
- `diffusion_model.py`
  - Diffusion proxy from velocity and depth.

## Runtime Use

- `simulation/domain.py` uses `get_width_range(...)`.
- `simulation/run_simulation.py` uses `get_climate_multiplier(...)`.

## Notes

- Some estimator modules are available for extension/research workflows and are not directly called by the current Streamlit path.
