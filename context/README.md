# context

## Purpose

Provides lightweight context-mapping helpers that convert user text into terrain-aware categories used elsewhere in the simulation pipeline.

## Files

- `river_classifier.py`
  - `classify_river(river_type_input)`: normalizes user input to lowercase.
- `location_mapper.py`
  - `map_location_to_region(location)`: keyword-based mapping to one of:
    - `mountain`
    - `plain`
    - `urban`
    - `coastal`

## Notes

- This folder currently contains rule-based helpers and no external dependencies.
- The main Streamlit app currently uses direct UI selection for terrain, so these helpers are optional extension points.
