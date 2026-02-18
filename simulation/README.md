# simulation

## Purpose

Implements spatial domain creation, PINN runtime inference orchestration, accumulation metrics, and validation studies.

## Files

- `domain.py`
  - `latlon_to_local_xy(...)`: converts geographic coordinates into local normalized coordinates.
  - `create_spatial_domain(...)`: builds `x/y` arrays using terrain width ranges.
- `run_simulation.py`
  - `run_simulation(x, y, season, time_value)`:
    - loads the PINN model
    - evaluates concentration over spatial mesh at a time slice
    - scales output by seasonal climate factor
- `accumulation_metrics.py`
  - `compute_accumulation(frames)`: mean accumulation over time frames.
- `scenario_runner.py`
  - Legacy alternate runner using `config.py` grid settings and iterative evaluation.
- `validation_suite.py`
  - CFL and monotonicity stability diagnostics for explicit FDM baseline
  - PINN vs FDM error metrics (`MSE`, `RMSE`, `L2`, relative `L2`)
  - Discrete PDE residual diagnostics
  - Grid refinement study
  - Diffusion robustness and domain-scaling robustness utilities
  - Runtime benchmarking and training metadata loader
- `uncertainty_quantification.py`
  - Conformal uncertainty calibration against reference solver errors
  - Reliability curve generation (nominal vs empirical coverage)
  - Local quantile half-width uncertainty maps
- `observational_validation.py`
  - Transect CSV normalization
  - Observation-vs-prediction metrics (`RMSE`, `MAE`, `R2`, `NSE`, bias)
  - Bootstrap confidence intervals and interval coverage checks
- `ablation_study.py`
  - Lightweight training ablations:
    - full (data + PDE + BC/IC)
    - no PDE
    - no BC/IC
    - data-only
  - Reports hold-out and grid-level metrics for scientific contribution claims

## Input/Output Contract

- Inputs: domain vectors (`x`, `y`), `season`, `time`.
- Output: 2D concentration field as `numpy.ndarray`.

## Notes

- Model loading is cached in `core/pinn_inference.py` (`lru_cache`) so repeated time-slice evaluations do not reload weights.
