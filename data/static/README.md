# data/static

## Purpose

Holds small versioned JSON configuration artifacts used directly by runtime modules.

## Files

- `river_width_classes.json`
  - Terrain-wise width ranges used by `simulation/domain.py`.
- `climate_scaling.json`
  - Seasonal multipliers applied in `simulation/run_simulation.py`.
- `manning_coefficients.json`
  - Reserved for Manning-based depth/roughness extensions.

## Source of Truth

Values are typically derived offline via `data_validation/analysis/` and then promoted here after review.

## Version Control

These files should be committed because they are required for reproducible runtime behavior.
