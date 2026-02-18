# simulation/scenario_runner.py

import torch
import numpy as np
import config

def run_simulation(pinn, width, length):
    x = torch.linspace(0, length, config.NX)
    y = torch.linspace(0, width, config.NY)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    frames = []
    times = np.linspace(0, config.TIME_MAX, config.N_TIME_STEPS)

    for t in times:
        t_tensor = torch.full_like(X, t)
        C = pinn.evaluate(
            X.reshape(-1,1),
            Y.reshape(-1,1),
            t_tensor.reshape(-1,1)
        )
        frames.append(C.reshape(config.NX, config.NY).numpy())

    return frames, X.numpy(), Y.numpy()
