import os
import pandas as pd
from datetime import datetime

BASE_OUT = "outputs"
IMG_DIR = os.path.join(BASE_OUT, "images")
ANIM_DIR = os.path.join(BASE_OUT, "animations")
TAB_DIR = os.path.join(BASE_OUT, "tables")

def ensure_dirs():
    for d in [BASE_OUT, IMG_DIR, ANIM_DIR, TAB_DIR]:
        os.makedirs(d, exist_ok=True)

def get_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_table(data: dict, run_id: str):
    df = pd.DataFrame([data])
    path = os.path.join(TAB_DIR, f"run_{run_id}.csv")
    df.to_csv(path, index=False)
    return path
