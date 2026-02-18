import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def _resolve_column(df, candidates):
    lower_cols = {c.lower(): c for c in df.columns}
    for key in candidates:
        for low, raw in lower_cols.items():
            if key in low:
                return raw
    return None


def normalize_observation_table(df):
    if df is None or len(df) == 0:
        raise ValueError("Observation table is empty.")

    dist_col = _resolve_column(df, ["distance", "dist", "chainage", "x"])
    conc_col = _resolve_column(df, ["obs", "concentration", "conc", "value"])
    time_col = _resolve_column(df, ["time", "t"])

    if dist_col is None or conc_col is None:
        raise ValueError(
            "CSV must include distance and observed concentration columns. "
            "Examples: `distance_m`, `concentration_obs`, optional `time`."
        )

    out = pd.DataFrame(
        {
            "distance": pd.to_numeric(df[dist_col], errors="coerce"),
            "obs": pd.to_numeric(df[conc_col], errors="coerce"),
        }
    )

    if time_col is not None:
        out["time"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        out["time"] = np.nan

    out = out.dropna(subset=["distance", "obs"]).copy()
    if len(out) < 3:
        raise ValueError("At least 3 valid observation rows are required.")
    return out


def _compute_metrics(obs, pred):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - obs
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    nse = 1.0 - ss_res / (ss_tot + 1e-12)
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "bias": float(bias),
        "r2": float(r2),
        "nse": float(nse),
    }


def bootstrap_metric_ci(obs, pred, n_boot=500, ci=95, seed=2026):
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    n = len(obs)
    rng = np.random.default_rng(seed)

    stats = {"rmse": [], "mae": [], "bias": [], "r2": [], "nse": []}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        m = _compute_metrics(obs[idx], pred[idx])
        for k in stats:
            stats[k].append(m[k])

    lo_q = (100 - ci) / 2
    hi_q = 100 - lo_q
    out = {}
    for k, vals in stats.items():
        out[k] = {
            "mean": float(np.mean(vals)),
            "lo": float(np.percentile(vals, lo_q)),
            "hi": float(np.percentile(vals, hi_q)),
        }
    return out


def evaluate_transect_observations(
    observations_df,
    x,
    y,
    times,
    predicted_frames_xy,
    q_interval_halfwidth=None,
):
    obs_df = normalize_observation_table(observations_df).copy()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(times, dtype=float)
    frames = np.asarray(predicted_frames_xy, dtype=float)

    if frames.shape != (len(t), len(x), len(y)):
        raise ValueError(
            "Predicted frames shape must be (T, NX, NY): "
            f"got {frames.shape}, expected {(len(t), len(x), len(y))}."
        )

    # Assume transect along centerline in y, projected by distance.
    y_mid = float(0.5 * (y[0] + y[-1]))
    d = obs_df["distance"].to_numpy(dtype=float)
    d_min = float(np.min(d))
    d_max = float(np.max(d))
    if abs(d_max - d_min) < 1e-12:
        x_query = np.full_like(d, 0.5 * (x[0] + x[-1]))
    else:
        x_query = x[0] + (d - d_min) / (d_max - d_min) * (x[-1] - x[0])

    time_values = obs_df["time"].to_numpy(dtype=float)
    time_default = float(t[-1])
    time_query = np.where(np.isfinite(time_values), time_values, time_default)
    time_query = np.clip(time_query, t[0], t[-1])

    interp = RegularGridInterpolator(
        (t, x, y),
        frames,
        bounds_error=False,
        fill_value=None,
    )
    pts = np.column_stack([time_query, x_query, np.full_like(x_query, y_mid)])
    pred = interp(pts)
    obs = obs_df["obs"].to_numpy(dtype=float)

    metrics = _compute_metrics(obs, pred)
    ci = bootstrap_metric_ci(obs, pred, n_boot=500, ci=95)

    coverage = None
    if q_interval_halfwidth is not None:
        q = float(max(q_interval_halfwidth, 0.0))
        coverage = float(np.mean(np.abs(pred - obs) <= q))

    out_df = obs_df.copy()
    out_df["pred"] = pred
    out_df["abs_error"] = np.abs(pred - obs)

    if q_interval_halfwidth is not None:
        out_df["pred_lo"] = pred - q
        out_df["pred_hi"] = pred + q

    return {
        "point_metrics": metrics,
        "bootstrap_ci": ci,
        "coverage_with_interval": coverage,
        "comparison_df": out_df.sort_values("distance").reset_index(drop=True),
    }
