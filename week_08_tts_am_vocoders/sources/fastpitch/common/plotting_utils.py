import matplotlib.pylab as plt
import numpy as np


def save_figure_to_numpy(fig):
    fig.canvas.draw()

    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    buf = buf.reshape(height, width, 4)

    buf = buf[:, :, [1, 2, 3]]

    plt.close(fig)
    return buf

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    return save_figure_to_numpy(fig)
