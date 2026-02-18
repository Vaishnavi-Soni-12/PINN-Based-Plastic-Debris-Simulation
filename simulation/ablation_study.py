import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import RegularGridInterpolator

from simulation.validation_suite import (
    compute_discrete_pde_residual,
    compute_error_timeseries,
)


class SmallPINN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, xyz):
        return self.net(xyz)


def _sample_supervised_points(x, y, times, frames_xy, n_samples, rng):
    interp = RegularGridInterpolator((times, x, y), frames_xy, bounds_error=False, fill_value=None)

    t_s = rng.uniform(times[0], times[-1], n_samples)
    x_s = rng.uniform(x[0], x[-1], n_samples)
    y_s = rng.uniform(y[0], y[-1], n_samples)
    pts = np.column_stack([t_s, x_s, y_s])
    target = interp(pts)
    xyz = np.column_stack([x_s, y_s, t_s])
    return xyz, target.reshape(-1, 1)


def _sample_collocation_points(x, y, times, n_samples, rng):
    t_s = rng.uniform(times[0], times[-1], n_samples)
    x_s = rng.uniform(x[0], x[-1], n_samples)
    y_s = rng.uniform(y[0], y[-1], n_samples)
    return np.column_stack([x_s, y_s, t_s])


def _sample_bc_ic_points(x, y, times, frames_xy, n_samples, rng):
    # IC points
    x_ic = rng.uniform(x[0], x[-1], n_samples)
    y_ic = rng.uniform(y[0], y[-1], n_samples)
    t_ic = np.full(n_samples, times[0])
    ic_xyz = np.column_stack([x_ic, y_ic, t_ic])

    interp_ic = RegularGridInterpolator(
        (times, x, y),
        frames_xy,
        bounds_error=False,
        fill_value=None,
    )
    ic_target = interp_ic(np.column_stack([t_ic, x_ic, y_ic])).reshape(-1, 1)

    # BC points for Neumann (zero gradient normal)
    n_edge = max(1, n_samples // 4)
    t_bc = rng.uniform(times[0], times[-1], n_edge)

    y_l = rng.uniform(y[0], y[-1], n_edge)
    x_l = np.full(n_edge, x[0])
    left = np.column_stack([x_l, y_l, t_bc])

    y_r = rng.uniform(y[0], y[-1], n_edge)
    x_r = np.full(n_edge, x[-1])
    right = np.column_stack([x_r, y_r, t_bc])

    x_b = rng.uniform(x[0], x[-1], n_edge)
    y_b = np.full(n_edge, y[0])
    bottom = np.column_stack([x_b, y_b, t_bc])

    x_t = rng.uniform(x[0], x[-1], n_edge)
    y_t = np.full(n_edge, y[-1])
    top = np.column_stack([x_t, y_t, t_bc])

    return ic_xyz, ic_target, left, right, bottom, top


def _pde_residual(model, xyz, u, v, diffusion):
    xyz = xyz.requires_grad_(True)
    c = model(xyz)
    grads = torch.autograd.grad(
        c, xyz, grad_outputs=torch.ones_like(c), create_graph=True
    )[0]
    cx = grads[:, 0:1]
    cy = grads[:, 1:2]
    ct = grads[:, 2:3]

    cxx = torch.autograd.grad(
        cx, xyz, grad_outputs=torch.ones_like(cx), create_graph=True
    )[0][:, 0:1]
    cyy = torch.autograd.grad(
        cy, xyz, grad_outputs=torch.ones_like(cy), create_graph=True
    )[0][:, 1:2]
    return ct + u * cx + v * cy - diffusion * (cxx + cyy)


def _neumann_bc_loss(model, left, right, bottom, top):
    def grad_xy(inp):
        inp = inp.requires_grad_(True)
        c = model(inp)
        return torch.autograd.grad(
            c, inp, grad_outputs=torch.ones_like(c), create_graph=True
        )[0]

    g_l = grad_xy(left)[:, 0:1]
    g_r = grad_xy(right)[:, 0:1]
    g_b = grad_xy(bottom)[:, 1:2]
    g_t = grad_xy(top)[:, 1:2]
    return (
        torch.mean(g_l ** 2)
        + torch.mean(g_r ** 2)
        + torch.mean(g_b ** 2)
        + torch.mean(g_t ** 2)
    )


def _predict_frames(model, x, y, times, batch_size=8192):
    xg, yg = np.meshgrid(x, y, indexing="ij")
    flat_xy = np.column_stack([xg.ravel(), yg.ravel()])
    out = []
    model.eval()
    with torch.no_grad():
        for t in times:
            t_col = np.full((flat_xy.shape[0], 1), float(t))
            xyz = np.column_stack([flat_xy, t_col])
            pred_chunks = []
            for i in range(0, xyz.shape[0], batch_size):
                chunk = torch.tensor(
                    xyz[i : i + batch_size], dtype=torch.float32
                )
                pred_chunks.append(model(chunk).cpu().numpy())
            pred = np.vstack(pred_chunks).reshape(len(x), len(y))
            out.append(pred)
    return np.array(out, dtype=float)


def run_ablation_experiment(
    x,
    y,
    times,
    fdm_frames_xy,
    u,
    v,
    diffusion,
    epochs=80,
    n_data=2000,
    n_collocation=2000,
    n_bc_ic=1200,
    seed=2026,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    xyz_data_np, target_data_np = _sample_supervised_points(
        x, y, times, fdm_frames_xy, n_data, rng
    )
    xyz_val_np, target_val_np = _sample_supervised_points(
        x, y, times, fdm_frames_xy, max(400, n_data // 5), rng
    )
    xyz_col_np = _sample_collocation_points(x, y, times, n_collocation, rng)
    ic_xyz_np, ic_target_np, left_np, right_np, bottom_np, top_np = _sample_bc_ic_points(
        x, y, times, fdm_frames_xy, n_bc_ic, rng
    )

    xyz_data = torch.tensor(xyz_data_np, dtype=torch.float32)
    target_data = torch.tensor(target_data_np, dtype=torch.float32)
    xyz_val = torch.tensor(xyz_val_np, dtype=torch.float32)
    target_val = torch.tensor(target_val_np, dtype=torch.float32)
    xyz_col = torch.tensor(xyz_col_np, dtype=torch.float32)
    ic_xyz = torch.tensor(ic_xyz_np, dtype=torch.float32)
    ic_target = torch.tensor(ic_target_np, dtype=torch.float32)
    left = torch.tensor(left_np, dtype=torch.float32)
    right = torch.tensor(right_np, dtype=torch.float32)
    bottom = torch.tensor(bottom_np, dtype=torch.float32)
    top = torch.tensor(top_np, dtype=torch.float32)

    variants = {
        "full": {"w_pde": 1.0, "w_bcic": 1.0},
        "no_pde": {"w_pde": 0.0, "w_bcic": 1.0},
        "no_bcic": {"w_pde": 1.0, "w_bcic": 0.0},
        "data_only": {"w_pde": 0.0, "w_bcic": 0.0},
    }

    histories = {}
    summary = []

    dx = (x[-1] - x[0]) / max(len(x) - 1, 1)
    dy = (y[-1] - y[0]) / max(len(y) - 1, 1)

    for name, cfg in variants.items():
        model = SmallPINN(width=32)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        mse = nn.MSELoss()

        hist = {
            "epoch": [],
            "total_loss": [],
            "data_loss": [],
            "pde_loss": [],
            "bcic_loss": [],
            "val_rmse": [],
        }

        for ep in range(1, int(epochs) + 1):
            model.train()
            opt.zero_grad()

            pred_data = model(xyz_data)
            loss_data = mse(pred_data, target_data)

            res = _pde_residual(model, xyz_col, u=u, v=v, diffusion=diffusion)
            loss_pde = torch.mean(res ** 2)

            pred_ic = model(ic_xyz)
            loss_ic = mse(pred_ic, ic_target)
            loss_bc = _neumann_bc_loss(model, left, right, bottom, top)
            loss_bcic = loss_ic + loss_bc

            total = loss_data + cfg["w_pde"] * loss_pde + cfg["w_bcic"] * loss_bcic
            total.backward()
            opt.step()

            with torch.no_grad():
                val_pred = model(xyz_val)
                val_rmse = torch.sqrt(mse(val_pred, target_val))

            hist["epoch"].append(ep)
            hist["total_loss"].append(float(total.item()))
            hist["data_loss"].append(float(loss_data.item()))
            hist["pde_loss"].append(float(loss_pde.item()))
            hist["bcic_loss"].append(float(loss_bcic.item()))
            hist["val_rmse"].append(float(val_rmse.item()))

        pred_frames = _predict_frames(model, x, y, times)
        err = compute_error_timeseries(pred_frames, fdm_frames_xy)
        res_stats = compute_discrete_pde_residual(
            frames=pred_frames,
            times=times,
            dx=dx,
            dy=dy,
            u=u,
            v=v,
            diffusion=diffusion,
        )
        summary.append(
            {
                "variant": name,
                "final_val_rmse": hist["val_rmse"][-1],
                "final_grid_rmse": err["rmse_final"],
                "final_relative_l2": err["relative_l2_final"],
                "final_pde_residual_mse": res_stats["residual_mse_final"],
            }
        )
        histories[name] = hist

    return {"summary": summary, "histories": histories}
