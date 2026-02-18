# core/pinn_inference.py
import torch
import os
from functools import lru_cache
from core.pinn_model import PINN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "pinn_weights.pt")

@lru_cache(maxsize=1)
def load_pinn():
    model = PINN()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()
    return model
