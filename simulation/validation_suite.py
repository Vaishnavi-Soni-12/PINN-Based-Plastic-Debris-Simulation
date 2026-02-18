import json
import os
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _to_xy_field(field, x, y):
    arr = np.asarray(field, dtype=float)
    if arr.shape == (len(x), len(y)):
        return arr, False
    if arr.shape == (len(y), len(x)):
        return arr.T, True
    raise ValueError(
        "Field shape does not match x/y lengths: "
        f"{arr.shape} vs ({len(x)}, {len(y)}) or ({len(y)}, {len(x)})."
    )


def stable_dt_bound(dx, dy, u, v, diffusion, safety=0.9):
    adv_denom = abs(u) / max(dx, 1e-12) + abs(v) / max(dy, 1e-12)
    diff_denom = 2.0 * diffusion * (
        1.0 / max(dx, 1e-12) ** 2 + 1.0 / max(dy, 1e-12) ** 2
    )
    dt_adv = np.inf if adv_denom <= 0 else 1.0 / adv_denom
    dt_diff = np.inf if diff_denom <= 0 else 1.0 / diff_denom
    return safety * min(dt_adv, dt_diff)


def compute_stability_diagnostics(dx, dy, dt, u, v, diffusion):
    cfl_x = abs(u) * dt / max(dx, 1e-12)
    cfl_y = abs(v) * dt / max(dy, 1e-12)
    adv_cfl = cfl_x + cfl_y
    diff_cfl = 2.0 * diffusion * dt * (
        1.0 / max(dx, 1e-12) ** 2 + 1.0 / max(dy, 1e-12) ** 2
    )
    center_coeff = 1.0 - adv_cfl - diff_cfl
    margin = min(1.0 - adv_cfl, 1.0 - diff_cfl, center_coeff)
    return {
        "dx": float(dx),
        "dy": float(dy),
        "dt": float(dt),
        "cfl_x": float(cfl_x),
        "cfl_y": float(cfl_y),
        "adv_cfl_total": float(adv_cfl),
        "diff_cfl_total": float(diff_cfl),
        "center_coefficient": float(center_coeff),
        "stability_margin": float(margin),
        "is_stable_by_cfl": bool(adv_cfl < 1.0 and diff_cfl < 1.0),
        "is_monotone": bool(center_coeff >= 0.0),
    }


def _fdm_step(prev, dt, dx, dy, u, v, diffusion):
    nxt = prev.copy()
    c = prev[1:-1, 1:-1]

    if u >= 0:
        dcdx = (c - prev[:-2, 1:-1]) / dx
    else:
        dcdx = (prev[2:, 1:-1] - c) / dx

    if v >= 0:
        dcdy = (c - prev[1:-1, :-2]) / dy
    else:
        dcdy = (prev[1:-1, 2:] - c) / dy

    lap_x = (prev[2:, 1:-1] - 2.0 * c + prev[:-2, 1:-1]) / (dx ** 2)
    lap_y = (prev[1:-1, 2:] - 2.0 * c + prev[1:-1, :-2]) / (dy ** 2)

    nxt[1:-1, 1:-1] = c + dt * (-u * dcdx - v * dcdy + diffusion * (lap_x + lap_y))

    # Zero-gradient (no-flux) boundary treatment.
    nxt[0, :] = nxt[1, :]
    nxt[-1, :] = nxt[-2, :]
    nxt[:, 0] = nxt[:, 1]
    nxt[:, -1] = nxt[:, -2]
    return nxt


def run_fdm_reference(x, y, times, initial_field, u, v, diffusion, safety=0.9):
    nx = len(x)
    ny = len(y)
    n_times = len(times)
    dx = (x[-1] - x[0]) / max(nx - 1, 1)
    dy = (y[-1] - y[0]) / max(ny - 1, 1)
    dt_limit = stable_dt_bound(dx, dy, u, v, diffusion, safety=safety)
    initial_xy, was_transposed = _to_xy_field(initial_field, x, y)

    frames_xy = np.zeros((n_times, nx, ny), dtype=float)
    state = initial_xy.copy()
    frames_xy[0] = state

    dt_used = []
    substeps = []

    for k in range(1, n_times):
        interval = float(times[k] - times[k - 1])
        if interval <= 0:
            frames_xy[k] = state
            continue

        if np.isfinite(dt_limit) and dt_limit > 0:
            n_sub = max(1, int(np.ceil(interval / dt_limit)))
        else:
            n_sub = 1
        dt = interval / n_sub

        for _ in range(n_sub):
            state = _fdm_step(state, dt, dx, dy, u, v, diffusion)

        frames_xy[k] = state
        dt_used.append(dt)
        substeps.append(n_sub)

    dt_report = max(dt_used) if dt_used else 0.0
    stability = compute_stability_diagnostics(dx, dy, dt_report, u, v, diffusion)
    stability["dt_limit"] = None if not np.isfinite(dt_limit) else float(dt_limit)
    stability["max_substeps_per_interval"] = int(max(substeps)) if substeps else 1
    if was_transposed:
        frames = np.transpose(frames_xy, (0, 2, 1))
    else:
        frames = frames_xy
    return frames, stability


def compute_error_timeseries(pinn_frames, fdm_frames):
    diff = pinn_frames - fdm_frames
    mse = np.mean(diff ** 2, axis=(1, 2))
    rmse = np.sqrt(mse)
    l2 = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    fdm_l2 = np.linalg.norm(fdm_frames.reshape(fdm_frames.shape[0], -1), axis=1)
    relative_l2 = l2 / (fdm_l2 + 1e-12)
    return {
        "mse": mse,
        "rmse": rmse,
        "l2": l2,
        "relative_l2": relative_l2,
        "signed_final_error": diff[-1],
        "abs_final_error": np.abs(diff[-1]),
        "rmse_final": float(rmse[-1]),
        "l2_final": float(l2[-1]),
        "relative_l2_final": float(relative_l2[-1]),
    }


def compute_mass_metrics(frames):
    mass = np.sum(frames, axis=(1, 2))
    denom = abs(mass[0]) + 1e-12
    relative_drift = (mass - mass[0]) / denom
    return {
        "mass": mass,
        "relative_drift": relative_drift,
        "max_abs_relative_drift": float(np.max(np.abs(relative_drift))),
    }


def compute_discrete_pde_residual(frames, times, dx, dy, u, v, diffusion):
    edge_order_xy = 2 if min(frames.shape[1], frames.shape[2]) >= 3 else 1
    edge_order_t = 2 if len(times) >= 3 else 1

    ct_all = np.gradient(frames, times, axis=0, edge_order=edge_order_t)
    residual_maps = np.zeros_like(frames, dtype=float)

    for k in range(len(times)):
        c = frames[k]
        cx = np.gradient(c, dx, axis=0, edge_order=edge_order_xy)
        cy = np.gradient(c, dy, axis=1, edge_order=edge_order_xy)
        cxx = np.gradient(cx, dx, axis=0, edge_order=edge_order_xy)
        cyy = np.gradient(cy, dy, axis=1, edge_order=edge_order_xy)
        residual_maps[k] = ct_all[k] + u * cx + v * cy - diffusion * (cxx + cyy)

    core = residual_maps[:, 1:-1, 1:-1]
    residual_mse = np.mean(core ** 2, axis=(1, 2))
    return {
        "residual_maps": residual_maps,
        "residual_mse": residual_mse,
        "residual_mse_final": float(residual_mse[-1]),
    }


def _resample_field(src_field, src_x, src_y, dst_x, dst_y):
    interp = RegularGridInterpolator(
        (src_x, src_y),
        src_field,
        bounds_error=False,
        fill_value=None,
    )
    xg, yg = np.meshgrid(dst_x, dst_y, indexing="ij")
    pts = np.column_stack((xg.ravel(), yg.ravel()))
    return interp(pts).reshape(len(dst_x), len(dst_y))


def run_grid_refinement_study(
    base_x,
    base_y,
    times,
    initial_field,
    u,
    v,
    diffusion,
    factors=(0.5, 1.0, 1.5),
):
    init_xy, _ = _to_xy_field(initial_field, base_x, base_y)
    runs = []
    for factor in sorted(set(factors)):
        nx = max(20, int(round(len(base_x) * factor)))
        ny = max(12, int(round(len(base_y) * factor)))
        x = np.linspace(base_x[0], base_x[-1], nx)
        y = np.linspace(base_y[0], base_y[-1], ny)
        init = _resample_field(init_xy, base_x, base_y, x, y)
        frames, stability = run_fdm_reference(
            x, y, times, init, u=u, v=v, diffusion=diffusion
        )
        final_xy, _ = _to_xy_field(frames[-1], x, y)
        dx = (x[-1] - x[0]) / max(nx - 1, 1)
        dy = (y[-1] - y[0]) / max(ny - 1, 1)
        runs.append(
            {
                "factor": float(factor),
                "nx": int(nx),
                "ny": int(ny),
                "x": x,
                "y": y,
                "final": final_xy,
                "h": float(max(dx, dy)),
                "stability_margin": float(stability["stability_margin"]),
            }
        )

    ref_idx = int(np.argmax([r["nx"] * r["ny"] for r in runs]))
    ref = runs[ref_idx]
    ref_final = ref["final"]

    hs = []
    errors = []
    for r in runs:
        resampled = _resample_field(r["final"], r["x"], r["y"], ref["x"], ref["y"])
        rmse = np.sqrt(np.mean((resampled - ref_final) ** 2))
        r["rmse_vs_reference"] = float(rmse)
        hs.append(r["h"])
        errors.append(rmse)

    hs = np.asarray(hs)
    errors = np.asarray(errors)
    valid = errors > 0
    observed_order = np.nan
    if np.count_nonzero(valid) >= 2:
        coeff = np.polyfit(np.log(hs[valid]), np.log(errors[valid]), deg=1)
        observed_order = float(coeff[0])

    return {
        "runs": runs,
        "reference_nx": int(ref["nx"]),
        "reference_ny": int(ref["ny"]),
        "observed_order": observed_order,
    }


def _normalize(field):
    shifted = field - np.min(field)
    scale = np.max(shifted) + 1e-12
    return shifted / scale


def _corrcoef_flat(a, b):
    a_flat = a.ravel()
    b_flat = b.ravel()
    if np.std(a_flat) < 1e-12 or np.std(b_flat) < 1e-12:
        return 1.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _hotspot_jaccard(a, b, q=0.9):
    a_thr = np.quantile(a, q)
    b_thr = np.quantile(b, q)
    ma = a >= a_thr
    mb = b >= b_thr
    union = np.count_nonzero(ma | mb)
    if union == 0:
        return 1.0
    inter = np.count_nonzero(ma & mb)
    return float(inter / union)


def run_diffusion_robustness(
    x,
    y,
    times,
    initial_field,
    u,
    v,
    base_diffusion,
    multipliers=(0.5, 0.75, 1.0, 1.25, 1.5),
):
    init_xy, _ = _to_xy_field(initial_field, x, y)
    results = []
    for m in multipliers:
        diffusion = max(1e-10, base_diffusion * m)
        frames, _ = run_fdm_reference(
            x, y, times, init_xy, u=u, v=v, diffusion=diffusion
        )
        final, _ = _to_xy_field(frames[-1], x, y)
        results.append(
            {"multiplier": float(m), "diffusion": float(diffusion), "final": final}
        )

    baseline = min(results, key=lambda r: abs(r["multiplier"] - 1.0))
    base_norm = _normalize(baseline["final"])

    for r in results:
        norm = _normalize(r["final"])
        r["structural_corr"] = _corrcoef_flat(base_norm, norm)
        r["hotspot_overlap"] = _hotspot_jaccard(base_norm, norm)

    return {"results": results}


def run_domain_scaling_robustness(
    x,
    y,
    times,
    season,
    run_pinn_slice_fn,
    scales=(0.75, 1.0, 1.25),
):
    x0 = x[0]
    y0 = y[0]
    results = []

    for scale in scales:
        xs = x0 + (x - x0) * scale
        ys = y0 + (y - y0) * scale
        frames = np.array(
            [run_pinn_slice_fn(xs, ys, season, float(t)) for t in times],
            dtype=float,
        )
        accumulation = np.mean(frames, axis=0)
        results.append(
            {
                "scale": float(scale),
                "accumulation": accumulation,
                "mass_drift": compute_mass_metrics(frames)["max_abs_relative_drift"],
            }
        )

    baseline = min(results, key=lambda r: abs(r["scale"] - 1.0))
    base_norm = _normalize(baseline["accumulation"])

    for r in results:
        norm = _normalize(r["accumulation"])
        r["structural_corr"] = _corrcoef_flat(base_norm, norm)
        r["hotspot_overlap"] = _hotspot_jaccard(base_norm, norm)

    return {"results": results}


def benchmark_runtime(pinn_callable, fdm_callable):
    t0 = perf_counter()
    pinn_output = pinn_callable()
    pinn_seconds = perf_counter() - t0

    t1 = perf_counter()
    fdm_output = fdm_callable()
    fdm_seconds = perf_counter() - t1

    speedup = fdm_seconds / (pinn_seconds + 1e-12)
    return {
        "pinn_output": pinn_output,
        "fdm_output": fdm_output,
        "pinn_seconds": float(pinn_seconds),
        "fdm_seconds": float(fdm_seconds),
        "speedup_fdm_over_pinn": float(speedup),
    }


def load_training_metadata(path="core/training_metadata.json"):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
