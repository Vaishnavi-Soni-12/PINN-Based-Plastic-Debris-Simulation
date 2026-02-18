# visualization/animation.py

import streamlit as st
import matplotlib.pyplot as plt
import config

def animate(frames):
    placeholder = st.empty()
    for i, frame in enumerate(frames):
        fig, ax = plt.subplots()
        ax.imshow(frame, cmap=config.COLORMAP, origin="lower")
        ax.set_title(f"Time step {i}")
        placeholder.pyplot(fig)
        plt.close(fig)
