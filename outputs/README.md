# outputs

## Purpose

Holds generated runtime artifacts from Streamlit simulation runs.

## Subfolders

- `images/`: PNG visual summaries and validation plots.
- `animations/`: MP4 concentration evolution videos.
- `tables/`: CSV run metadata and key summary metrics.

Advanced validation save action additionally writes:
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
- `journal_observation_benchmark_*.png` (when observation CSV is provided)
- `journal_ablation_performance_*.png` and `journal_ablation_loss_*.png` (when ablation is run)
- `journal_stress_matrix_rmse_*.png` (when stress matrix is run)
- `run_journal_validation_*.csv` (in `tables/`)
- `transect_benchmark_*.csv` (in `tables/`, optional)
- `ablation_summary_*.csv` (in `tables/`, optional)
- `stress_matrix_*.csv` (in `tables/`, optional)

## Generation Source

Artifacts are created through:

- `utils/output_manager.py`
- `utils/save_outputs.py`
- save actions in `app.py`

## Version Control Policy

This folder is runtime-generated and excluded from git commits. Only folder-level README files are tracked.
