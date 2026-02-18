# context/location_mapper.py

def map_location_to_region(location: str):
    """
    Very lightweight abstraction.
    """
    location = location.lower()

    if any(k in location for k in ["himalaya", "alps", "mountain"]):
        return "mountain"
    elif any(k in location for k in ["city", "urban"]):
        return "urban"
    elif any(k in location for k in ["coastal", "delta"]):
        return "coastal"
    else:
        return "plain"
