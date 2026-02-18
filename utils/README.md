# utils

## Purpose

Provides utility functions for geocoding, output directory management, and persistent export of figures/animations/tables.

## Files

- `place_resolver.py`
  - Resolves place names to `latitude/longitude` via Nominatim.
  - Maintains local cache file `geocode_cache.csv` to avoid repeated API calls.
  - Recovers automatically from malformed or empty cache file.
- `output_manager.py`
  - Creates output directories.
  - Generates timestamp-based run IDs.
  - Saves tabular run summaries to CSV.
- `save_outputs.py`
  - Saves static figures (PNG).
  - Saves time-evolution animations (MP4 via `ffmpeg`).
- `geocode_cache.csv`
  - Local runtime artifact; excluded from version control.

## Notes

- If geocoding is unavailable and a place is not cached, `resolve_place(...)` raises a runtime error.
