# hydrology/velocity_model.py

def compute_velocity(discharge, width, depth):
    return discharge / (width * depth)
