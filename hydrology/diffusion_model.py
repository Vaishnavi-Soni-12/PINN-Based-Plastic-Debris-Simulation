# hydrology/diffusion_model.py

def compute_diffusion(velocity, depth, beta=0.1):
    return beta * velocity * depth
