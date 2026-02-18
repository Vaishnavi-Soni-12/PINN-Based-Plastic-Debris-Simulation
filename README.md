# Physics-Informed Neural Network Framework for Riverine Plastic Transport Analysis

This project provides an interactive Streamlit system to simulate, visualize, and analyze riverine plastic transport using a Physics-Informed Neural Network (PINN) surrogate.

## Project Makers (Equal Contribution)

| Member | Title | Contribution Share |
| --- | --- | --- |
| Vaishnavi Soni | Co-Lead Hydrology and Validation Architect | 50% |
| Zian Rajeshkumar Surani | Co-Lead PINN and Simulation Systems Architect | 50% |

## What This Project Does

- Accepts place name, terrain class, season, and simulation horizon from the UI.
- Resolves location coordinates through Nominatim with local CSV caching.
- Builds a location-aware 2D spatial domain using terrain-specific width ranges.
- Runs PINN inference over space and time to estimate concentration fields.
- Computes accumulation and reviewer-focused validation metrics against an explicit FDM baseline.
- Quantifies time-resolved errors (`MSE`, `RMSE`, `L2`), PDE residual consistency, and mass drift.
- Runs FDM grid-refinement, perturbation robustness, and runtime speed-up analysis.
- Calibrates uncertainty intervals with conformal reliability diagnostics.
- Supports optional field transect CSV benchmarking with bootstrap confidence intervals.
- Includes training ablation and multi-scenario stress matrix studies for reviewer defense.
- Saves figures, animations, and run-level tabular summaries for reproducibility.

## End-to-End Flow

1. User configures simulation in `app.py`.
2. `utils/place_resolver.py` resolves `lat/lon` from place name.
3. `simulation/domain.py` generates `x/y` coordinates using width ranges from `data/static/river_width_classes.json`.
4. `simulation/run_simulation.py` loads model weights from `core/pinn_weights.pt` and infers concentration fields.
5. `simulation/accumulation_metrics.py` computes accumulation maps from time frames.
6. `simulation/validation_suite.py` computes advanced numerical validation diagnostics.
7. `simulation/uncertainty_quantification.py` calibrates uncertainty and reliability metrics.
8. `simulation/observational_validation.py` benchmarks against optional transect observations.
9. `simulation/ablation_study.py` runs lightweight physics ablation experiments.
10. `app.py` renders concentration, accumulation, temporal, and validation plots.
11. `utils/output_manager.py` and `utils/save_outputs.py` save outputs to `outputs/`.
12. `data_validation/` scripts can be run offline to re-derive static constants from source datasets.

## Repository Structure

- `context/` lightweight region and river-type mapping helpers.
- `core/` PINN model definition, weight loading, and static config loader.
- `data/` static constants and local raw reference datasets.
- `data_validation/` offline scripts for deriving and validating constants.
- `hydrology/` hydrology parameter and estimator utilities.
- `simulation/` domain creation and PINN runtime execution logic.
- `utils/` place resolution, output path management, and save helpers.
- `visualization/` reusable plotting and animation helpers.
- `outputs/` generated figures, animations, and summary tables.

Each major folder includes its own `README.md` with file-level details.

## Setup

### Prerequisites

- Python 3.10+ recommended
- `ffmpeg` installed and available in PATH (needed for MP4 export)

### Install

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Run the App

```powershell
streamlit run app.py
```

## Offline Data Validation Workflow

Use `data_validation/` only when updating static constants:

```powershell
cd data_validation\analysis
python width_statistics.py
python chirps_validation.py
python slope_statistics.py
python discharge_statistics.py
```

Then review JSON outputs in `data_validation/outputs/` and copy approved values into `data/static/`.

## Journal-Defense Validation Protocol

Section 8 of `app.py` now includes a reviewer-facing validation layer:

- CFL-stable explicit FDM baseline with no-flux boundary handling
- Explicit stability diagnostics beyond CFL:
  - advective CFL
  - diffusive CFL
  - center-coefficient monotonicity check
  - stability margin and adaptive sub-stepping report
- Time-resolved error metrics:
  - `MSE vs time`
  - `RMSE vs time`
  - final-frame `L2` and relative `L2`
- Spatial signed error heatmap:
  - `PINN - FDM` at final time
- Convergence-style diagnostics:
  - PINN data loss curve vs PINN PDE residual curve (log scale)
  - FDM PDE residual shown as control baseline
- Physical consistency:
  - mass conservation curves and relative drift
- Numerical depth:
  - FDM grid refinement study with observed order estimate
- Robustness:
  - diffusion perturbation structural similarity
  - domain scaling structural persistence
- Computational analysis:
  - PINN runtime vs FDM runtime
  - speed-up factor
  - optional training-time metadata read from `core/training_metadata.json`

Additional defense sections:

- Section 9: uncertainty quantification and reliability calibration
  - conformal intervals
  - local uncertainty half-width map
  - nominal vs empirical coverage
- Section 10: optional real-data transect benchmark
  - CSV ingestion and metric reporting with bootstrap CIs
- Section 11: training ablation study
  - full vs no-PDE vs no-BC/IC vs data-only
- Section 12: reviewer stress matrix
  - all terrain-season combinations with mean/worst-case summary

## Output Artifacts

- `outputs/images/` PNG visualizations
- `outputs/animations/` MP4 evolution clips
- `outputs/tables/` CSV run metadata and summary metrics

Advanced validation images include:
- `journal_validation_comparison_*.png`
- `journal_mse_vs_time_*.png`
- `journal_loss_vs_residual_*.png`
- `journal_mass_conservation_*.png`
- `journal_mass_drift_*.png`
- `journal_grid_refinement_*.png`
- `journal_diffusion_robustness_*.png`
- `journal_domain_scaling_robustness_*.png`
- `journal_uq_reliability_*.png`
- `journal_uq_halfwidth_map_*.png`
- `journal_observation_benchmark_*.png` (if observation CSV provided)
- `journal_ablation_performance_*.png` (if ablation run)
- `journal_ablation_loss_*.png` (if ablation run)
- `journal_stress_matrix_rmse_*.png` (if stress matrix run)

File names use timestamp-based run IDs (`YYYYMMDD_HHMMSS`) for traceability.

## Files and Folders to Exclude from Git Commits

Do not commit local/generated/large artifacts:

- `venv/` (local environment)
- `__pycache__/` and `*.pyc`
- `outputs/` runtime-generated artifacts
- `utils/geocode_cache.csv` local geocode cache
- `data/cache/` local temporary cache
- `data/chirps/` large raw NetCDF datasets
- `data/grwl/` raw GRWL shapefile datasets
- `data/hydrosheds/` raw HydroSHEDS rasters
- `data_validation/outputs/` regenerated offline validation outputs
- `.vscode/` machine-specific editor settings

These exclusions are also encoded in `.gitignore`.

## Recommended Commit Scope

Commit source and small static configs:

- Python code under `context/`, `core/`, `hydrology/`, `simulation/`, `utils/`, `visualization/`
- `app.py`, `config.py`, `requirements.txt`
- `data/static/*.json`
- Documentation (`README.md` files)
- `.gitignore`

## Figure Gallery (`images/`)

### `1.png` App Landing Page
Main UI view showing project context and simulation controls.
![1.png](images/1.png)

### `2.png` Cross-Sectional Accumulation Profile
Lateral transect profile used to inspect centerline/bank accumulation tendency.
![2.png](images/2.png)

### `3.png` Mean Concentration vs Time
Time evolution of domain-mean concentration for trend interpretation.
![3.png](images/3.png)

### `4.png` Final Concentration Field
PINN-predicted concentration map at the final simulation time.
![4.png](images/4.png)

### `5.png` Accumulated Plastic Density
Time-aggregated accumulation heatmap highlighting persistent hotspots.
![5.png](images/5.png)

### `6.png` Normalized Accumulation Map
Scaled accumulation view for cross-scenario comparability.
![6.png](images/6.png)

### `7.png` Spatial Variance Over Time
Tracks heterogeneity in concentration distribution across time.
![7.png](images/7.png)

### `8.png` Advanced Validation Section (Legacy UI View)
Section overview screenshot showing validation narrative in the interface.
![8.png](images/8.png)

### `9.png` Seasonal Sensitivity Indicator
Bar comparison of dry/normal/wet scaling behavior.
![9.png](images/9.png)

### `8.1.png` Stability Diagnostics Panel
Advective/diffusive CFL and stability margin with explicit solver checks.
![8.1.png](images/8.1.png)

### `8.2.png` PINN vs FDM with Signed Error Heatmap
Side-by-side final fields and signed spatial error (`PINN - FDM`).
![8.2.png](images/8.2.png)

### `8.3.png` Time-Resolved Error Curve
`MSE` and `RMSE` evolution across simulation time.
![8.3.png](images/8.3.png)

### `8.4.png` Loss vs PDE Residual Convergence Plot
Log-scale comparison of data loss and PDE residual behavior.
![8.4.png](images/8.4.png)

### `8.5.png` FDM Grid Refinement Study
Error against finest grid and observed convergence order.
![8.5.png](images/8.5.png)

### `8.6.png` Diffusion Robustness Similarity
Structural correlation and hotspot overlap under diffusion perturbation.
![8.6.png](images/8.6.png)

### `8.7.1.png` Relative Mass Drift
Mass drift (%) comparison between PINN and FDM over time.
![8.7.1.png](images/8.7.1.png)

### `8.7.2.png` Mass Conservation Comparison
Total mass trajectories for PINN and FDM baselines.
![8.7.2.png](images/8.7.2.png)

### `8.7.3.png` Computational Analysis Panel
Runtime/speed-up metrics and scientific framing section snapshot.
![8.7.3.png](images/8.7.3.png)

### `8.8.png` Domain Scaling Robustness
Similarity metrics under domain-scale perturbation.
![8.8.png](images/8.8.png)

### `8.8.2.png` Reliability Calibration Curve
Nominal vs empirical coverage with ideal-calibration reference.
![8.8.2.png](images/8.8.2.png)

### `8.8.3.png` Local Uncertainty Map
Spatial map of local 90% interval half-width.
![8.8.3.png](images/8.8.3.png)

### `10.1.png` Optional Real-Data Benchmark UI
CSV-based transect upload and benchmarking workflow panel.
![10.1.png](images/10.1.png)

### `11.png` Temporal L2 Error (Legacy Plot)
Legacy time-wise L2 error curve retained as supplemental reference.
![11.png](images/11.png)

### `12.png` Mass Comparison (Legacy Plot)
Legacy mass comparison chart retained for completeness.
![12.png](images/12.png)

### `13.png` Diffusion Sensitivity Curve (Legacy Plot)
Legacy final-mean concentration response to diffusion coefficient.
![13.png](images/13.png)
