# fetch_grwl.py
# Offline script to load GRWL river width data
# Requires manual download of GRWL dataset (CSV/Shapefile)

import pandas as pd

def load_grwl(filepath):
    return pd.read_csv(filepath)

if __name__ == "__main__":
    print("Provide path to GRWL dataset manually and run statistics scripts.")
