import numpy as np
import pandas as pd


def _split_time_indices(n_times, calibration_fraction=0.5):
    cut = int(np.floor(n_times * calibration_fraction))
    cut = min(max(cut, 1), n_times - 1)
    return np.arange(0, cut), np.arange(cut, n_times)


def conformal_uq_from_reference(
    pinn_frames,
    reference_frames,
    levels=(0.5, 0.8, 0.9, 0.95),
    calibration_fraction=0.5,
):
    pinn = np.asarray(pinn_frames, dtype=float)
    ref = np.asarray(reference_frames, dtype=float)
    if pinn.shape != ref.shape:
        raise ValueError(f"Shape mismatch: PINN {pinn.shape}, reference {ref.shape}.")

    calib_idx, test_idx = _split_time_indices(pinn.shape[0], calibration_fraction)
    abs_err = np.abs(pinn - ref)
    calib_abs_err = abs_err[calib_idx]
    test_abs_err = abs_err[test_idx]

    reliability_rows = []
    for level in levels:
        q = np.quantile(calib_abs_err.reshape(-1), level)
        coverage = np.mean(test_abs_err.reshape(-1) <= q)
        reliability_rows.append(
            {
                "nominal_coverage": float(level),
                "empirical_coverage": float(coverage),
                "quantile_half_width": float(q),
            }
        )

    reliability_df = pd.DataFrame(reliability_rows)

    q90_global = float(np.quantile(calib_abs_err.reshape(-1), 0.90))
    q90_map = np.quantile(calib_abs_err, 0.90, axis=0)
    final_mean = pinn[-1]
    final_lower_global = final_mean - q90_global
    final_upper_global = final_mean + q90_global
    final_lower_local = final_mean - q90_map
    final_upper_local = final_mean + q90_map

    local_coverage_90 = np.mean(test_abs_err <= q90_map[None, :, :])

    return {
        "reliability_df": reliability_df,
        "q90_global": q90_global,
        "q90_map": q90_map,
        "local_coverage_90": float(local_coverage_90),
        "final_lower_global": final_lower_global,
        "final_upper_global": final_upper_global,
        "final_lower_local": final_lower_local,
        "final_upper_local": final_upper_local,
        "calibration_samples": int(calib_abs_err.size),
        "test_samples": int(test_abs_err.size),
    }
