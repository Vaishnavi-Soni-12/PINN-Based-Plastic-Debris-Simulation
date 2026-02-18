# visualization

## Purpose

Contains reusable visualization helpers for concentration heatmaps and frame-by-frame animations.

## Files

- `heatmap.py`
  - `plot_heatmap(field, title)` using shared colormap from `config.py`.
- `animation.py`
  - `animate(frames)` to render sequential frames in Streamlit.
- `plots.py`
  - Currently empty placeholder for future shared plotting utilities.

## Notes

- The main app also contains inline plotting logic in `app.py`.
- This folder can be expanded to centralize all visualization rendering in one place.
