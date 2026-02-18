# core

## Purpose

Contains the PINN model architecture, model weight loading, and static config loading used by hydrology and simulation modules.

## Files

- `pinn_model.py`
  - Defines `PINN`, a fully connected network:
    - Input: `(x, y, t)` (3 features)
    - Hidden: 3 layers, 64 units each, `Tanh`
    - Output: concentration scalar
- `pinn_inference.py`
  - Loads weights from `core/pinn_weights.pt`
  - Exposes `load_pinn()` for inference mode execution
- `config_loader.py`
  - Reads JSON files from `data/static/`
  - Exposes:
    - `RIVER_WIDTH_CLASSES`
    - `CLIMATE_SCALING`
- `pinn_weights.pt`
  - Trained model weights used at runtime
- `training_metadata.json`
  - Optional metadata file for reporting training evidence in validation outputs
  - Recommended keys:
    - `training_time_seconds`
    - `epochs`
    - `optimizer`
    - `final_training_loss`
    - `loss_curve_image_path`

## Runtime Dependency

`simulation/run_simulation.py` calls `load_pinn()` from this folder for each simulation time step request.
`load_pinn()` is cached (`lru_cache`) so weights are loaded once per process.

## Caution

If you retrain the model, replace `pinn_weights.pt` with a compatible state dict shape, otherwise loading will fail.
