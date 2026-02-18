import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def save_figure(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_animation(frames, path):
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], origin="lower", aspect="auto")

    def update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=300
    )

    ani.save(path, writer="ffmpeg", dpi=200)
    plt.close(fig)
