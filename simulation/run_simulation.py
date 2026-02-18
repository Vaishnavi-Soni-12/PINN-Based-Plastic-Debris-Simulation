import torch
import numpy as np
from core.pinn_inference import load_pinn
from hydrology.river_parameters import get_climate_multiplier

def run_simulation(x, y, season, time_value):
    model = load_pinn()
    climate_factor = get_climate_multiplier(season)

    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, time_value)

    coords = np.stack(
        [X.flatten(), Y.flatten(), T.flatten()],
        axis=1
    )

    coords_t = torch.tensor(coords, dtype=torch.float32)

    with torch.no_grad():
        concentration = model(coords_t).numpy().reshape(X.shape)

    return concentration * climate_factor
