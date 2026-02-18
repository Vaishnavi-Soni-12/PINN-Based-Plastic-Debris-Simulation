import numpy as np

def compute_accumulation(frames):
    return np.mean(frames, axis=0)
