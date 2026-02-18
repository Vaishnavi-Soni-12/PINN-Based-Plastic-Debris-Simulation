import os
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut

CACHE_PATH = os.path.join(os.path.dirname(__file__), "geocode_cache.csv")

_geolocator = Nominatim(
    user_agent="pinn_river_sim",
    timeout=10
)

def _ensure_cache():
    """
    Ensure cache file exists and has valid headers.
    """
    if not os.path.exists(CACHE_PATH):
        df = pd.DataFrame(columns=["place", "latitude", "longitude"])
        df.to_csv(CACHE_PATH, index=False)
        return df

    # File exists but may be empty or corrupted
    try:
        df = pd.read_csv(CACHE_PATH)
        if set(df.columns) != {"place", "latitude", "longitude"}:
            raise ValueError("Invalid cache schema")
        return df
    except Exception:
        # Reset corrupted or empty file
        df = pd.DataFrame(columns=["place", "latitude", "longitude"])
        df.to_csv(CACHE_PATH, index=False)
        return df

def resolve_place(place_name: str):
    place_key = place_name.strip().lower()

    # ---------- LOAD / FIX CACHE ----------
    cache = _ensure_cache()

    hit = cache[cache["place"] == place_key]
    if not hit.empty:
        return float(hit.iloc[0]["latitude"]), float(hit.iloc[0]["longitude"])

    # ---------- ONLINE GEOCODING ----------
    try:
        location = _geolocator.geocode(place_name)
        if location:
            lat, lon = location.latitude, location.longitude

            new_row = pd.DataFrame([{
                "place": place_key,
                "latitude": lat,
                "longitude": lon
            }])

            cache = pd.concat([cache, new_row], ignore_index=True)
            cache.to_csv(CACHE_PATH, index=False)

            return lat, lon

    except (GeocoderUnavailable, GeocoderTimedOut):
        pass

    # ---------- HARD FAIL (EXPLICIT, CLEAN) ----------
    raise RuntimeError(
        "Geocoding unavailable and place not present in local cache."
    )
