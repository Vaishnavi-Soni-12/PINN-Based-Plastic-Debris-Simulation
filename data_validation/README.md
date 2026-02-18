# data_validation

## Purpose

Offline-only pipeline for deriving and validating hydrology constants from global reference datasets. This folder is not imported by the Streamlit runtime path.

## What It Produces

- Width class ranges from GRWL
- Climate seasonal multipliers from CHIRPS
- Placeholder slope and discharge classes for calibration

Outputs are written to `data_validation/outputs/`, then manually reviewed and promoted to `data/static/`.

## Folder Layout

- `fetch/`: placeholder dataset loader scripts and fetch notes.
- `analysis/`: scripts that compute statistical constants.
- `outputs/`: generated JSON files from analysis scripts.

## Standard Workflow

1. Ensure local datasets are available in `data/`.
2. Run analysis scripts from the `analysis/` folder.
3. Inspect generated JSON in `data_validation/outputs/`.
4. Copy approved values into `data/static/`.
5. Re-run app and verify behavior.

## Typical Commands

```powershell
cd data_validation\analysis
python width_statistics.py
python chirps_validation.py
python slope_statistics.py
python discharge_statistics.py
```

## Notes

- `fetch/*.py` files are lightweight placeholders to document expected source data handling.
- `data_validation/outputs/` is generated content and should usually not be committed.

## Journal Submission Note

- Reviewer-facing numerical validation (FDM benchmarking, PDE residual diagnostics, grid refinement, robustness, runtime analysis) is executed in the runtime validation layer (`app.py` + `simulation/validation_suite.py`), not in this offline folder.
- This folder remains focused on provenance and calibration constants that feed the simulation assumptions.
