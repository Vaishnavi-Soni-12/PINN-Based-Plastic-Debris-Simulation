# visualization/heatmap.py

import matplotlib.pyplot as plt
import config

def plot_heatmap(field, title):
    fig, ax = plt.subplots()
    im = ax.imshow(field, cmap=config.COLORMAP, origin="lower")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return fig
