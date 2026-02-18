import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "data", "static")

def load_json(filename):
    path = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing static config: {filename}")
    with open(path, "r") as f:
        return json.load(f)

# Public accessors
RIVER_WIDTH_CLASSES = load_json("river_width_classes.json")
CLIMATE_SCALING = load_json("climate_scaling.json")
