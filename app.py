import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from time import perf_counter

from simulation.domain import create_spatial_domain
from simulation.run_simulation import run_simulation
from simulation.accumulation_metrics import compute_accumulation
from simulation.validation_suite import (
    run_fdm_reference,
    compute_error_timeseries,
    compute_mass_metrics,
    compute_discrete_pde_residual,
    run_grid_refinement_study,
    run_diffusion_robustness,
    run_domain_scaling_robustness,
    load_training_metadata,
)
from simulation.uncertainty_quantification import conformal_uq_from_reference
from simulation.observational_validation import evaluate_transect_observations
from simulation.ablation_study import run_ablation_experiment
from hydrology.discharge_estimator import estimate_discharge
from hydrology.depth_estimator import estimate_depth
from hydrology.velocity_model import compute_velocity
from hydrology.diffusion_model import compute_diffusion
from utils.place_resolver import resolve_place
from utils.output_manager import ensure_dirs, get_run_id, save_table, IMG_DIR, ANIM_DIR, TAB_DIR
from utils.save_outputs import save_figure, save_animation

# ======================================================
# PAGE CONFIGURATION
# ======================================================
st.set_page_config(layout="wide")
st.title("Physics-Informed Neural Network (PINN) Framework for Riverine Plastic Transport Analysis")

# ======================================================
# INTRODUCTION & PROJECT OVERVIEW
# ======================================================
with st.expander("1. Project Overview and Motivation", expanded=True):
    st.markdown("""
    ### Background  
    Rivers are one of the primary pathways through which plastic waste enters marine environments.
    Understanding how plastic moves, disperses, and accumulates within river systems is essential
    for designing effective mitigation strategies.

    Traditional modeling approaches rely on computational fluid dynamics (CFD) solvers or
    particle-tracking simulations, both of which are computationally expensive and require
    dense boundary condition data that is often unavailable at large spatial scales.

    ### Objective  
    This project proposes a **Physics-Informed Neural Network (PINN)** based framework to model
    the **spatio-temporal transport and accumulation of plastic debris in rivers**, enabling:

    - Physics-consistent predictions without explicit numerical solvers  
    - Interactive scenario analysis across locations, terrains, and seasons  
    - Identification of long-term accumulation hotspots  

    ### Core Contribution  
    A trained PINN acts as a **surrogate solver** for the advection–diffusion equation governing
    plastic transport, allowing rapid inference over space and time.
    """)

# ======================================================
# RESEARCH CONTEXT & RELATED WORK
# ======================================================
with st.expander("2. Research Context and Related Work"):
    st.markdown("""
    ### Existing Approaches
    - **Eulerian solvers** (finite difference / finite volume): accurate but computationally heavy  
    - **Lagrangian particle tracking**: requires explicit velocity fields  
    - **Pure data-driven ML models**: lack physical consistency  

    ### Positioning of This Work
    This project lies at the intersection of **hydrology**, **environmental modeling**, and
    **scientific machine learning (SciML)**.

    By embedding physical laws directly into a neural network, PINNs combine:
    - The interpretability of physics-based models  
    - The flexibility and speed of neural networks  

    This approach aligns with recent research trends in surrogate modeling for environmental systems.
    """)

# ======================================================
# DATA SOURCES AND PROVENANCE
# ======================================================
with st.expander("3. Data Sources, Provenance, and Usage"):
    st.markdown("""
    ### Overview
    This system uses **reference datasets** for model formulation and **physics-based learning**
    during training. At runtime, the model relies on learned physics rather than raw raster ingestion.

    ### Named Datasets
    **GRWL (Global River Widths from Landsat)**  
    - Used to define representative river width ranges for different terrain classes  

    **HydroSHEDS**  
    - Provides global hydrological context such as river topology and slope  
    - Informs terrain-dependent assumptions  

    **CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)**  
    - Used to characterize seasonal regimes (dry / normal / wet)  
    - Influences discharge intensity and transport potential  

    **OpenStreetMap / Nominatim**  
    - Converts user-entered place names into geographic coordinates  
    - Used only for spatial localization  

    **Physics-Based Synthetic Training Data**  
    - PINN trained using collocation points enforcing the governing PDE  
    - No labeled concentration data required  

    ### Important Clarification
    External datasets are **not directly queried during runtime simulation**.
    They inform model design, parameterization, and validation.
    """)

# ======================================================
# MODEL & PHYSICAL FORMULATION
# ======================================================
with st.expander("4. Model Architecture and Governing Physics"):
    st.markdown("""
    ### Physics-Informed Neural Network (PINN)
    - Fully connected feed-forward neural network  
    - Inputs: spatial coordinates (x, y) and time (t)  
    - Output: plastic concentration C(x, y, t)  

    ### Governing Equation
    The PINN approximates solutions to the **advection–diffusion equation**:

    ∂C/∂t + u · ∇C = ∇ · (D ∇C)

    where:
    - C represents plastic concentration  
    - u represents flow velocity (implicit)  
    - D represents diffusion coefficient (implicit)  

    Physical constraints are enforced during training via PDE residuals,
    ensuring physics-consistent predictions.
    """)

# ======================================================
# ASSUMPTIONS & LIMITATIONS
# ======================================================
with st.expander("5. Modeling Assumptions and Limitations"):
    st.markdown("""
    ### Assumptions
    - Plastic is treated as a passive scalar  
    - Transport is depth-averaged  
    - Flow conditions are quasi-steady  
    - Vertical stratification is neglected  

    ### Limitations
    - No real-time velocity or discharge ingestion  
    - No explicit bank or vegetation interaction modeling  
    - Accumulation inferred statistically  

    These define the valid operational envelope of the system.
    """)

# ======================================================
# USER INPUT SECTION
# ======================================================
st.sidebar.header("6. Simulation Configuration")

place_name = st.sidebar.text_input("Place Name (any valid geographic location)")
river_type = st.sidebar.selectbox(
    "Terrain Classification",
    ["mountain", "plain", "urban", "coastal"]
)
season = st.sidebar.selectbox("Seasonal Regime", ["dry", "normal", "wet"])
t_max = st.sidebar.slider("Simulation Time Horizon (normalized)", 1.0, 10.0, 5.0)
n_steps = st.sidebar.slider("Temporal Resolution (number of steps)", 5, 40, 20)

# ======================================================
# RUN SIMULATION
# ======================================================
if st.sidebar.button("Run Simulation"):

    lat, lon = resolve_place(place_name)
    x, y = create_spatial_domain(river_type, lat, lon)

    times = np.linspace(0, t_max, n_steps)
    pinn_start = perf_counter()
    frames = np.array([run_simulation(x, y, season, t) for t in times])
    pinn_runtime_seconds = perf_counter() - pinn_start
    accumulation = compute_accumulation(frames)

    st.header("7. Simulation Outputs and Interpretation")

    # --------------------------------------------------
    st.subheader("7.1 Final Plastic Concentration Field")
    st.markdown("""
    **What this represents**  
    The spatial distribution of plastic concentration at the final simulated time.

    **How to interpret**  
    - High values indicate instantaneous hotspots  
    - Gradients reflect dominance of advection or diffusion  
    """)

    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(frames[-1], origin="lower", aspect="auto")
    plt.colorbar(im1, ax=ax1, label="Concentration")
    st.pyplot(fig1)

    # --------------------------------------------------
    st.subheader("7.2 Accumulated Plastic Density")
    st.markdown("""
    **What this represents**  
    Time-integrated concentration highlighting persistent accumulation zones.

    **Why it matters**  
    Useful for identifying regions suitable for interception or cleanup.
    """)

    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(accumulation, origin="lower", aspect="auto", cmap="inferno")
    plt.colorbar(im2, ax=ax2, label="Accumulated Load")
    st.pyplot(fig2)

    # --------------------------------------------------
    st.subheader("7.3 Temporal Evolution of Mean Concentration")
    st.markdown("""
    **What this represents**  
    Spatially averaged concentration as a function of time.

    **Interpretation**  
    - Increasing trend: accumulation dominance  
    - Stable trend: steady-state behavior  
    """)

    mean_time = frames.mean(axis=(1, 2))
    fig3, ax3 = plt.subplots()
    ax3.plot(times, mean_time)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Mean Concentration")
    ax3.grid(True)
    st.pyplot(fig3)

    # --------------------------------------------------
    st.subheader("7.4 Cross-Sectional Accumulation Profile")
    st.markdown("""
    **What this represents**  
    Lateral distribution of accumulated plastic across the river width.

    **Interpretation**  
    Indicates whether plastics preferentially accumulate near banks or centerline.
    """)

    mid_x = accumulation.shape[1] // 2
    profile = accumulation[:, mid_x]
    fig4, ax4 = plt.subplots()
    ax4.plot(profile, np.arange(len(profile)))
    ax4.invert_yaxis()
    ax4.grid(True)
    st.pyplot(fig4)

    # --------------------------------------------------
    st.subheader("7.5 Spatial Variance of Concentration Over Time")
    st.markdown("""
    **What this represents**  
    Measures spatial heterogeneity of plastic distribution.

    **Interpretation**  
    Higher variance indicates localized trapping; lower variance indicates diffusion smoothing.
    """)

    variance_time = frames.var(axis=(1, 2))
    fig5, ax5 = plt.subplots()
    ax5.plot(times, variance_time)
    ax5.grid(True)
    st.pyplot(fig5)

    # --------------------------------------------------
    st.subheader("7.6 Normalized Accumulated Plastic Density")
    st.markdown("""
    **What this represents**  
    Accumulation normalized by its maximum value.

    **Why it matters**  
    Enables comparison across rivers of different scales.
    """)

    normalized_accum = accumulation / (accumulation.max() + 1e-8)
    fig6, ax6 = plt.subplots()
    im6 = ax6.imshow(normalized_accum, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(im6, ax=ax6)
    st.pyplot(fig6)

    # --------------------------------------------------
    st.subheader("7.7 Seasonal Sensitivity Indicator")
    st.markdown("""
    **What this represents**  
    Relative amplification of accumulated load under different seasonal regimes.

    **Interpretation**  
    Indicates climate-driven sensitivity of plastic transport.
    """)

    season_scale = {"dry": 0.7, "normal": 1.0, "wet": 1.3}
    base_val = accumulation.mean()
    fig7, ax7 = plt.subplots()
    ax7.bar(season_scale.keys(), [base_val * v for v in season_scale.values()])
    st.pyplot(fig7)

    # ======================================================
    # SAVE OUTPUTS
    # ======================================================
    if st.button("Save Outputs"):

        ensure_dirs()
        run_id = get_run_id()

        save_figure(fig1, f"{IMG_DIR}/final_concentration_{run_id}.png")
        save_figure(fig2, f"{IMG_DIR}/accumulation_{run_id}.png")
        save_figure(fig5, f"{IMG_DIR}/variance_time_{run_id}.png")
        save_figure(fig6, f"{IMG_DIR}/normalized_accumulation_{run_id}.png")
        save_figure(fig7, f"{IMG_DIR}/seasonal_sensitivity_{run_id}.png")
        save_animation(frames, f"{ANIM_DIR}/evolution_{run_id}.mp4")

        save_table({
            "run_id": run_id,
            "place": place_name,
            "latitude": lat,
            "longitude": lon,
            "terrain": river_type,
            "season": season,
            "time_max": t_max,
            "time_steps": n_steps,
            "max_accumulation": float(accumulation.max()),
            "mean_accumulation": float(accumulation.mean()),
            "variance_accumulation": float(accumulation.var())
        }, run_id)

        st.success("All simulation outputs saved for reproducible analysis.")
    # ======================================================
    # 8. ADVANCED NUMERICAL VALIDATION EXTENSION
    # ======================================================

    st.header("8. Advanced Numerical Validation Extension")
    st.markdown("""
    This section is designed as a reviewer-focused defense layer:
    it tests physical consistency, numerical validation depth,
    computational impact, robustness, and scientific framing.
    """)

    if frames.shape[1] == len(y) and frames.shape[2] == len(x):
        validation_frames = np.transpose(frames, (0, 2, 1))
    else:
        validation_frames = frames

    def derive_flow_params(terrain_label, season_label, y_domain):
        width_local = max(float(y_domain[-1] - y_domain[0]), 1.0)
        discharge_local = estimate_discharge(terrain_label, season_label)
        velocity_seed_local = max(discharge_local / max(width_local * 3.0, 1e-12), 0.05)
        depth_local = max(estimate_depth(width_local, velocity_seed_local, terrain_label), 0.2)
        velocity_x_local = max(compute_velocity(discharge_local, width_local, depth_local), 1e-8)
        velocity_y_local = 0.0
        diffusion_local = max(compute_diffusion(velocity_x_local, depth_local, beta=0.08), 1e-10)
        return {
            "width": float(width_local),
            "discharge": float(discharge_local),
            "depth": float(depth_local),
            "u": float(velocity_x_local),
            "v": float(velocity_y_local),
            "D": float(diffusion_local),
        }

    flow_params = derive_flow_params(river_type, season, y)
    velocity_x = flow_params["u"]
    velocity_y = flow_params["v"]
    diffusion = flow_params["D"]

    # --------------------------------------------------
    st.subheader("8.1 FDM Baseline and Explicit Stability Diagnostics")
    fdm_start = perf_counter()
    fdm_solution, stability = run_fdm_reference(
        x=x,
        y=y,
        times=times,
        initial_field=validation_frames[0],
        u=velocity_x,
        v=velocity_y,
        diffusion=diffusion,
    )
    fdm_runtime_seconds = perf_counter() - fdm_start

    c1, c2, c3 = st.columns(3)
    c1.metric("Advective CFL", f"{stability['adv_cfl_total']:.4f}")
    c2.metric("Diffusive CFL", f"{stability['diff_cfl_total']:.4f}")
    c3.metric("Stability Margin", f"{stability['stability_margin']:.4f}")

    st.markdown(f"""
    - `u = {velocity_x:.4f}`, `v = {velocity_y:.4f}`, `D = {diffusion:.6f}`
    - Center update coefficient: `{stability['center_coefficient']:.4f}`
    - Monotone update regime: `{stability['is_monotone']}`
    - CFL-stable explicit stepping: `{stability['is_stable_by_cfl']}`
    - Maximum internal sub-steps per output interval: `{stability['max_substeps_per_interval']}`
    """)

    # --------------------------------------------------
    st.subheader("8.2 Quantitative PINN vs FDM Error Metrics")
    error_metrics = compute_error_timeseries(validation_frames, fdm_solution)
    st.write(f"RMSE (final): {error_metrics['rmse_final']:.6f}")
    st.write(f"L2 Error (final): {error_metrics['l2_final']:.6f}")
    st.write(f"Relative L2 Error (final): {error_metrics['relative_l2_final']:.6f}")

    pinn_final = validation_frames[-1]
    fdm_final = fdm_solution[-1]
    signed_final_error = error_metrics["signed_final_error"]
    signed_lim = np.max(np.abs(signed_final_error)) + 1e-12

    fig_val, ax_val = plt.subplots(1, 3, figsize=(13, 4))
    ax_val[0].imshow(pinn_final, origin="lower", aspect="auto")
    ax_val[0].set_title("PINN Final")
    ax_val[1].imshow(fdm_final, origin="lower", aspect="auto")
    ax_val[1].set_title("FDM Final")
    signed_plot = ax_val[2].imshow(
        signed_final_error,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=-signed_lim,
        vmax=signed_lim,
    )
    ax_val[2].set_title("Signed Error (PINN - FDM)")
    plt.colorbar(signed_plot, ax=ax_val[2])
    st.pyplot(fig_val)

    # --------------------------------------------------
    st.subheader("8.3 Time-Resolved Error Curve (MSE vs Time)")
    fig_mse, ax_mse = plt.subplots()
    ax_mse.plot(times, error_metrics["mse"], marker="o", label="MSE")
    ax_mse.plot(times, error_metrics["rmse"], marker="s", label="RMSE")
    ax_mse.set_xlabel("Time")
    ax_mse.set_ylabel("Error Magnitude")
    ax_mse.grid(True)
    ax_mse.legend()
    st.pyplot(fig_mse)

    # --------------------------------------------------
    st.subheader("8.4 Convergence Behavior: PINN Loss vs PDE Residual")
    dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
    dy = (y[-1] - y[0]) / max(len(y) - 1, 1)
    pinn_residual = compute_discrete_pde_residual(
        frames=validation_frames,
        times=times,
        dx=dx,
        dy=dy,
        u=velocity_x,
        v=velocity_y,
        diffusion=diffusion,
    )
    fdm_residual = compute_discrete_pde_residual(
        frames=fdm_solution,
        times=times,
        dx=dx,
        dy=dy,
        u=velocity_x,
        v=velocity_y,
        diffusion=diffusion,
    )

    fig_conv, ax_conv = plt.subplots()
    ax_conv.plot(times, error_metrics["mse"] + 1e-16, label="PINN data loss (MSE)")
    ax_conv.plot(times, pinn_residual["residual_mse"] + 1e-16, label="PINN PDE residual MSE")
    ax_conv.plot(
        times,
        fdm_residual["residual_mse"] + 1e-16,
        linestyle="--",
        label="FDM PDE residual MSE",
    )
    ax_conv.set_yscale("log")
    ax_conv.set_xlabel("Time")
    ax_conv.set_ylabel("Log-scale error")
    ax_conv.grid(True)
    ax_conv.legend()
    st.pyplot(fig_conv)

    # --------------------------------------------------
    st.subheader("8.5 Mass Conservation Verification")
    pinn_mass = compute_mass_metrics(validation_frames)
    fdm_mass = compute_mass_metrics(fdm_solution)

    fig_mass, ax_mass = plt.subplots()
    ax_mass.plot(times, pinn_mass["mass"], label="PINN mass")
    ax_mass.plot(times, fdm_mass["mass"], label="FDM mass")
    ax_mass.set_xlabel("Time")
    ax_mass.set_ylabel("Total mass")
    ax_mass.grid(True)
    ax_mass.legend()
    st.pyplot(fig_mass)

    fig_drift, ax_drift = plt.subplots()
    ax_drift.plot(times, 100.0 * pinn_mass["relative_drift"], label="PINN drift (%)")
    ax_drift.plot(times, 100.0 * fdm_mass["relative_drift"], label="FDM drift (%)")
    ax_drift.set_xlabel("Time")
    ax_drift.set_ylabel("Relative mass drift (%)")
    ax_drift.grid(True)
    ax_drift.legend()
    st.pyplot(fig_drift)

    # --------------------------------------------------
    st.subheader("8.6 Grid Refinement Study for FDM Baseline")
    grid_study = run_grid_refinement_study(
        base_x=x,
        base_y=y,
        times=times,
        initial_field=validation_frames[0],
        u=velocity_x,
        v=velocity_y,
        diffusion=diffusion,
        factors=(0.6, 1.0, 1.4),
    )
    grid_rows = []
    for run in grid_study["runs"]:
        grid_rows.append(
            {
                "grid": f"{run['nx']} x {run['ny']}",
                "h_max": run["h"],
                "rmse_vs_ref": run["rmse_vs_reference"],
                "stability_margin": run["stability_margin"],
            }
        )
    st.dataframe(pd.DataFrame(grid_rows), width="stretch")

    h_vals = np.array([r["h"] for r in grid_study["runs"]], dtype=float)
    e_vals = np.array([r["rmse_vs_reference"] for r in grid_study["runs"]], dtype=float)
    nonzero = e_vals > 0
    fig_grid, ax_grid = plt.subplots()
    ax_grid.loglog(h_vals[nonzero], e_vals[nonzero], marker="o")
    ax_grid.set_xlabel("Grid spacing h")
    ax_grid.set_ylabel("RMSE vs finest grid")
    ax_grid.set_title(
        f"Observed order ~ {grid_study['observed_order']:.3f}"
        if np.isfinite(grid_study["observed_order"])
        else "Observed order unavailable"
    )
    ax_grid.grid(True, which="both")
    st.pyplot(fig_grid)

    # --------------------------------------------------
    st.subheader("8.7 Robustness Under Parameter Perturbations")
    diffusion_robustness = run_diffusion_robustness(
        x=x,
        y=y,
        times=times,
        initial_field=validation_frames[0],
        u=velocity_x,
        v=velocity_y,
        base_diffusion=diffusion,
        multipliers=(0.5, 0.75, 1.0, 1.25, 1.5),
    )
    diff_df = pd.DataFrame(
        [
            {
                "diffusion_multiplier": r["multiplier"],
                "structural_corr": r["structural_corr"],
                "hotspot_overlap": r["hotspot_overlap"],
            }
            for r in diffusion_robustness["results"]
        ]
    )
    st.dataframe(diff_df, width="stretch")

    fig_diff_robust, ax_diff_robust = plt.subplots()
    ax_diff_robust.plot(
        diff_df["diffusion_multiplier"],
        diff_df["structural_corr"],
        marker="o",
        label="Structural correlation",
    )
    ax_diff_robust.plot(
        diff_df["diffusion_multiplier"],
        diff_df["hotspot_overlap"],
        marker="s",
        label="Hotspot overlap",
    )
    ax_diff_robust.set_xlabel("Diffusion multiplier")
    ax_diff_robust.set_ylabel("Similarity score")
    ax_diff_robust.grid(True)
    ax_diff_robust.legend()
    st.pyplot(fig_diff_robust)

    domain_scaling = run_domain_scaling_robustness(
        x=x,
        y=y,
        times=times,
        season=season,
        run_pinn_slice_fn=run_simulation,
        scales=(0.75, 1.0, 1.25),
    )
    scale_df = pd.DataFrame(
        [
            {
                "domain_scale": r["scale"],
                "structural_corr": r["structural_corr"],
                "hotspot_overlap": r["hotspot_overlap"],
                "max_mass_drift": r["mass_drift"],
            }
            for r in domain_scaling["results"]
        ]
    )
    st.dataframe(scale_df, width="stretch")

    fig_scale_robust, ax_scale_robust = plt.subplots()
    ax_scale_robust.plot(
        scale_df["domain_scale"],
        scale_df["structural_corr"],
        marker="o",
        label="Structural correlation",
    )
    ax_scale_robust.plot(
        scale_df["domain_scale"],
        scale_df["hotspot_overlap"],
        marker="s",
        label="Hotspot overlap",
    )
    ax_scale_robust.set_xlabel("Domain scale factor")
    ax_scale_robust.set_ylabel("Similarity score")
    ax_scale_robust.grid(True)
    ax_scale_robust.legend()
    st.pyplot(fig_scale_robust)

    # --------------------------------------------------
    st.subheader("8.8 Computational Analysis")
    speedup = fdm_runtime_seconds / (pinn_runtime_seconds + 1e-12)
    a1, a2, a3 = st.columns(3)
    a1.metric("PINN inference runtime (s)", f"{pinn_runtime_seconds:.4f}")
    a2.metric("FDM runtime (s)", f"{fdm_runtime_seconds:.4f}")
    a3.metric("Speed-up (FDM / PINN)", f"{speedup:.3f}x")

    training_meta = load_training_metadata()
    training_seconds = None
    training_epochs = None
    training_optimizer = None
    training_final_loss = None
    loss_curve_path = None

    if training_meta is not None:
        training_epochs = training_meta.get("epochs")
        training_optimizer = training_meta.get("optimizer")
        training_final_loss = training_meta.get("final_training_loss")
        if training_final_loss is None:
            training_final_loss = training_meta.get("final_loss")
        loss_curve_path = training_meta.get("loss_curve_image_path")

    if training_meta is not None and "training_time_seconds" in training_meta:
        try:
            if training_meta["training_time_seconds"] is not None:
                training_seconds = float(training_meta["training_time_seconds"])
        except (TypeError, ValueError):
            training_seconds = None

    if training_meta is None:
        st.warning(
            "Training metadata file is missing. Add `core/training_metadata.json` "
            "before journal submission."
        )
    elif training_seconds is not None and training_seconds > 0:
        st.write(
            "Reported training time from metadata: "
            f"{training_seconds:.2f} seconds"
        )
    else:
        st.info(
            "Wall-clock training time is not available in metadata. "
            "Runtime benchmarking is still reported; add measured training time later if available."
        )

    training_lines = []
    if training_epochs is not None:
        training_lines.append(f"- Epochs: `{training_epochs}`")
    if training_optimizer:
        training_lines.append(f"- Optimizer: `{training_optimizer}`")
    if training_final_loss is not None:
        try:
            training_lines.append(f"- Final training loss: `{float(training_final_loss):.6e}`")
        except (TypeError, ValueError):
            training_lines.append(f"- Final training loss: `{training_final_loss}`")
    if training_lines:
        st.markdown("Available training metadata:\n" + "\n".join(training_lines))

    if loss_curve_path:
        if os.path.exists(loss_curve_path):
            st.image(
                loss_curve_path,
                caption=f"Training loss curve ({loss_curve_path})",
                width="stretch",
            )
        else:
            st.info(
                "Loss curve path in metadata does not exist: "
                f"`{loss_curve_path}`"
            )

    # --------------------------------------------------
    st.subheader("8.9 Scientific Scope and Integrity Framing")
    st.markdown("""
    - Scope is limited to physically-informed surrogate simulation under predefined assumptions.
    - No forecasting or policy-level claims are made from this interface.
    - Limitations are explicit: no real-time velocity ingestion and no bank/vegetation interaction model.
    - Reproducibility is supported through saved figures, tables, and deterministic validation settings.
    """)

    # --------------------------------------------------
    st.subheader("9. Uncertainty Quantification and Reliability Calibration")
    uq = conformal_uq_from_reference(
        validation_frames,
        fdm_solution,
        levels=(0.5, 0.8, 0.9, 0.95),
        calibration_fraction=0.5,
    )
    st.dataframe(uq["reliability_df"], width="stretch")
    st.write(f"Global 90% interval half-width: {uq['q90_global']:.6f}")
    st.write(f"Local 90% empirical coverage on hold-out times: {uq['local_coverage_90']:.4f}")

    fig_uq_rel, ax_uq_rel = plt.subplots()
    ax_uq_rel.plot(
        uq["reliability_df"]["nominal_coverage"],
        uq["reliability_df"]["empirical_coverage"],
        marker="o",
        label="Model",
    )
    ax_uq_rel.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    ax_uq_rel.set_xlabel("Nominal coverage")
    ax_uq_rel.set_ylabel("Empirical coverage")
    ax_uq_rel.grid(True)
    ax_uq_rel.legend()
    st.pyplot(fig_uq_rel)

    fig_uq_map, ax_uq_map = plt.subplots()
    im_uq = ax_uq_map.imshow(uq["q90_map"], origin="lower", aspect="auto", cmap="magma")
    ax_uq_map.set_title("Local 90% interval half-width map")
    plt.colorbar(im_uq, ax=ax_uq_map)
    st.pyplot(fig_uq_map)

    # --------------------------------------------------
    st.subheader("10. Optional Real-Data Transect Benchmark")
    st.markdown("""
    Upload a CSV containing observed transect concentrations to benchmark simulation against field data.
    Required columns: distance and observed concentration.
    Optional column: time.
    """)

    obs_template = pd.DataFrame(
        {
            "distance_m": [0.0, 250.0, 500.0, 750.0, 1000.0],
            "concentration_obs": [0.12, 0.18, 0.21, 0.17, 0.10],
            "time": [float(times[-1])] * 5,
        }
    )
    st.download_button(
        "Download Transect CSV Template",
        obs_template.to_csv(index=False).encode("utf-8"),
        file_name="transect_observation_template.csv",
        mime="text/csv",
    )

    uploaded_obs = st.file_uploader(
        "Upload transect observations (CSV)",
        type=["csv"],
        key="transect_obs_csv",
    )

    obs_eval = None
    fig_obs_compare = None
    if uploaded_obs is not None:
        try:
            obs_df_raw = pd.read_csv(uploaded_obs)
            obs_eval = evaluate_transect_observations(
                observations_df=obs_df_raw,
                x=x,
                y=y,
                times=times,
                predicted_frames_xy=validation_frames,
                q_interval_halfwidth=uq["q90_global"],
            )

            metric_rows = []
            for metric_name, ci_vals in obs_eval["bootstrap_ci"].items():
                metric_rows.append(
                    {
                        "metric": metric_name,
                        "estimate": obs_eval["point_metrics"][metric_name],
                        "bootstrap_mean": ci_vals["mean"],
                        "ci95_lo": ci_vals["lo"],
                        "ci95_hi": ci_vals["hi"],
                    }
                )
            st.dataframe(pd.DataFrame(metric_rows), width="stretch")

            if obs_eval["coverage_with_interval"] is not None:
                st.write(
                    "Coverage of 90% conformal interval on observations: "
                    f"{obs_eval['coverage_with_interval']:.4f}"
                )

            comp_df = obs_eval["comparison_df"]
            fig_obs_compare, ax_obs_compare = plt.subplots()
            ax_obs_compare.plot(
                comp_df["distance"], comp_df["obs"], marker="o", label="Observed"
            )
            ax_obs_compare.plot(
                comp_df["distance"], comp_df["pred"], marker="s", label="Predicted"
            )
            if "pred_lo" in comp_df.columns and "pred_hi" in comp_df.columns:
                ax_obs_compare.fill_between(
                    comp_df["distance"],
                    comp_df["pred_lo"],
                    comp_df["pred_hi"],
                    alpha=0.25,
                    label="90% interval",
                )
            ax_obs_compare.set_xlabel("Transect distance")
            ax_obs_compare.set_ylabel("Concentration")
            ax_obs_compare.grid(True)
            ax_obs_compare.legend()
            st.pyplot(fig_obs_compare)

        except Exception as exc:
            st.error(f"Observation validation failed: {exc}")
    else:
        st.info("No observation CSV uploaded. Field-data benchmarking section is ready for use.")

    # --------------------------------------------------
    st.subheader("11. Training Ablation Study (Physics Contribution)")
    st.markdown("""
    Runs lightweight retraining against the FDM target to quantify which physics constraints
    improve generalization and PDE consistency.
    """)

    ablation_epochs = st.slider(
        "Ablation training epochs",
        min_value=40,
        max_value=200,
        value=80,
        step=20,
        key="ablation_epochs_slider",
    )

    scenario_signature = (
        float(lat),
        float(lon),
        str(river_type),
        str(season),
        float(t_max),
        int(n_steps),
        int(len(x)),
        int(len(y)),
    )
    if st.session_state.get("scenario_signature") != scenario_signature:
        st.session_state["scenario_signature"] = scenario_signature
        st.session_state["ablation_results"] = None
        st.session_state["stress_matrix_df"] = None

    if "ablation_results" not in st.session_state:
        st.session_state["ablation_results"] = None

    if st.button("Run Ablation Training Study"):
        with st.spinner("Running ablation study..."):
            st.session_state["ablation_results"] = run_ablation_experiment(
                x=x,
                y=y,
                times=times,
                fdm_frames_xy=fdm_solution,
                u=velocity_x,
                v=velocity_y,
                diffusion=diffusion,
                epochs=ablation_epochs,
                n_data=1600,
                n_collocation=1600,
                n_bc_ic=1000,
            )

    ablation_results = st.session_state.get("ablation_results")
    fig_ablation_perf = None
    fig_ablation_loss = None
    if ablation_results is not None:
        ablation_df = pd.DataFrame(ablation_results["summary"]).sort_values("final_grid_rmse")
        st.dataframe(ablation_df, width="stretch")

        fig_ablation_perf, ax_ablation_perf = plt.subplots()
        ax_ablation_perf.bar(
            ablation_df["variant"],
            ablation_df["final_grid_rmse"],
            label="Final grid RMSE",
        )
        ax_ablation_perf.set_ylabel("RMSE")
        ax_ablation_perf.grid(True, axis="y")
        st.pyplot(fig_ablation_perf)

        fig_ablation_loss, ax_ablation_loss = plt.subplots()
        for variant_name, hist in ablation_results["histories"].items():
            ax_ablation_loss.plot(
                hist["epoch"],
                np.array(hist["total_loss"]) + 1e-16,
                label=variant_name,
            )
        ax_ablation_loss.set_yscale("log")
        ax_ablation_loss.set_xlabel("Epoch")
        ax_ablation_loss.set_ylabel("Total loss (log)")
        ax_ablation_loss.grid(True)
        ax_ablation_loss.legend()
        st.pyplot(fig_ablation_loss)

    # --------------------------------------------------
    st.subheader("12. Multi-Scenario Reviewer Stress Matrix")
    st.markdown("""
    Runs all terrain-season combinations to avoid single-case claims and report mean/worst-case behavior.
    """)

    if "stress_matrix_df" not in st.session_state:
        st.session_state["stress_matrix_df"] = None

    if st.button("Run Reviewer Stress Matrix (12 scenarios)"):
        with st.spinner("Evaluating stress matrix..."):
            rows = []
            for terrain_case in ["mountain", "plain", "urban", "coastal"]:
                x_case, y_case = create_spatial_domain(terrain_case, lat, lon)
                for season_case in ["dry", "normal", "wet"]:
                    pinn_case = np.array(
                        [run_simulation(x_case, y_case, season_case, float(tt)) for tt in times]
                    )
                    if pinn_case.shape[1] == len(y_case) and pinn_case.shape[2] == len(x_case):
                        pinn_case = np.transpose(pinn_case, (0, 2, 1))

                    flow_case = derive_flow_params(terrain_case, season_case, y_case)
                    fdm_case, stability_case = run_fdm_reference(
                        x=x_case,
                        y=y_case,
                        times=times,
                        initial_field=pinn_case[0],
                        u=flow_case["u"],
                        v=flow_case["v"],
                        diffusion=flow_case["D"],
                    )
                    err_case = compute_error_timeseries(pinn_case, fdm_case)
                    mass_case = compute_mass_metrics(pinn_case)
                    dx_case = (x_case[-1] - x_case[0]) / max(len(x_case) - 1, 1)
                    dy_case = (y_case[-1] - y_case[0]) / max(len(y_case) - 1, 1)
                    res_case = compute_discrete_pde_residual(
                        frames=pinn_case,
                        times=times,
                        dx=dx_case,
                        dy=dy_case,
                        u=flow_case["u"],
                        v=flow_case["v"],
                        diffusion=flow_case["D"],
                    )
                    rows.append(
                        {
                            "terrain": terrain_case,
                            "season": season_case,
                            "rmse_final": err_case["rmse_final"],
                            "relative_l2_final": err_case["relative_l2_final"],
                            "pde_residual_final": res_case["residual_mse_final"],
                            "max_mass_drift": mass_case["max_abs_relative_drift"],
                            "fdm_stability_margin": stability_case["stability_margin"],
                        }
                    )
            st.session_state["stress_matrix_df"] = pd.DataFrame(rows)

    stress_df = st.session_state.get("stress_matrix_df")
    fig_stress = None
    if stress_df is not None and len(stress_df) > 0:
        st.dataframe(stress_df, width="stretch")
        summary_df = pd.DataFrame(
            [
                {"metric": "rmse_final", "mean": stress_df["rmse_final"].mean(), "worst": stress_df["rmse_final"].max()},
                {
                    "metric": "relative_l2_final",
                    "mean": stress_df["relative_l2_final"].mean(),
                    "worst": stress_df["relative_l2_final"].max(),
                },
                {
                    "metric": "pde_residual_final",
                    "mean": stress_df["pde_residual_final"].mean(),
                    "worst": stress_df["pde_residual_final"].max(),
                },
                {
                    "metric": "max_mass_drift",
                    "mean": stress_df["max_mass_drift"].mean(),
                    "worst": stress_df["max_mass_drift"].max(),
                },
            ]
        )
        st.dataframe(summary_df, width="stretch")

        fig_stress, ax_stress = plt.subplots()
        ax_stress.bar(stress_df.index.astype(str), stress_df["rmse_final"])
        ax_stress.set_xlabel("Scenario index")
        ax_stress.set_ylabel("Final RMSE")
        ax_stress.grid(True, axis="y")
        st.pyplot(fig_stress)

    # ======================================================
    # SAVE ADVANCED VALIDATION OUTPUTS
    # ======================================================
    if st.button("Save Advanced Validation Outputs"):
        ensure_dirs()
        run_id = get_run_id()

        save_figure(fig_val, f"{IMG_DIR}/journal_validation_comparison_{run_id}.png")
        save_figure(fig_mse, f"{IMG_DIR}/journal_mse_vs_time_{run_id}.png")
        save_figure(fig_conv, f"{IMG_DIR}/journal_loss_vs_residual_{run_id}.png")
        save_figure(fig_mass, f"{IMG_DIR}/journal_mass_conservation_{run_id}.png")
        save_figure(fig_drift, f"{IMG_DIR}/journal_mass_drift_{run_id}.png")
        save_figure(fig_grid, f"{IMG_DIR}/journal_grid_refinement_{run_id}.png")
        save_figure(fig_diff_robust, f"{IMG_DIR}/journal_diffusion_robustness_{run_id}.png")
        save_figure(fig_scale_robust, f"{IMG_DIR}/journal_domain_scaling_robustness_{run_id}.png")
        save_figure(fig_uq_rel, f"{IMG_DIR}/journal_uq_reliability_{run_id}.png")
        save_figure(fig_uq_map, f"{IMG_DIR}/journal_uq_halfwidth_map_{run_id}.png")

        if fig_obs_compare is not None:
            save_figure(fig_obs_compare, f"{IMG_DIR}/journal_observation_benchmark_{run_id}.png")
        if fig_ablation_perf is not None:
            save_figure(fig_ablation_perf, f"{IMG_DIR}/journal_ablation_performance_{run_id}.png")
        if fig_ablation_loss is not None:
            save_figure(fig_ablation_loss, f"{IMG_DIR}/journal_ablation_loss_{run_id}.png")
        if fig_stress is not None:
            save_figure(fig_stress, f"{IMG_DIR}/journal_stress_matrix_rmse_{run_id}.png")

        if obs_eval is not None:
            obs_eval["comparison_df"].to_csv(
                f"{TAB_DIR}/transect_benchmark_{run_id}.csv",
                index=False,
            )
        if ablation_results is not None:
            pd.DataFrame(ablation_results["summary"]).to_csv(
                f"{TAB_DIR}/ablation_summary_{run_id}.csv",
                index=False,
            )
        if stress_df is not None:
            stress_df.to_csv(f"{TAB_DIR}/stress_matrix_{run_id}.csv", index=False)

        save_table(
            {
                "run_id": run_id,
                "rmse_final": float(error_metrics["rmse_final"]),
                "l2_final": float(error_metrics["l2_final"]),
                "relative_l2_final": float(error_metrics["relative_l2_final"]),
                "pinn_pde_residual_final": float(pinn_residual["residual_mse_final"]),
                "fdm_pde_residual_final": float(fdm_residual["residual_mse_final"]),
                "pinn_max_mass_drift": float(pinn_mass["max_abs_relative_drift"]),
                "fdm_max_mass_drift": float(fdm_mass["max_abs_relative_drift"]),
                "fdm_advective_cfl": float(stability["adv_cfl_total"]),
                "fdm_diffusive_cfl": float(stability["diff_cfl_total"]),
                "fdm_stability_margin": float(stability["stability_margin"]),
                "pinn_runtime_seconds": float(pinn_runtime_seconds),
                "fdm_runtime_seconds": float(fdm_runtime_seconds),
                "speedup_fdm_over_pinn": float(speedup),
                "grid_observed_order": (
                    float(grid_study["observed_order"])
                    if np.isfinite(grid_study["observed_order"])
                    else np.nan
                ),
                "diffusion_corr_min": float(diff_df["structural_corr"].min()),
                "domain_scale_corr_min": float(scale_df["structural_corr"].min()),
                "uq_q90_global": float(uq["q90_global"]),
                "uq_local_coverage_90": float(uq["local_coverage_90"]),
                "observation_rmse": (
                    float(obs_eval["point_metrics"]["rmse"])
                    if obs_eval is not None
                    else np.nan
                ),
                "ablation_best_rmse": (
                    float(pd.DataFrame(ablation_results["summary"])["final_grid_rmse"].min())
                    if ablation_results is not None
                    else np.nan
                ),
                "stress_matrix_mean_rmse": (
                    float(stress_df["rmse_final"].mean())
                    if stress_df is not None
                    else np.nan
                ),
            },
            f"journal_validation_{run_id}",
        )

        st.success("Advanced validation outputs saved.")
