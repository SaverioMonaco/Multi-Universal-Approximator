"""Quick-look plots of a regression (the thesis figures are made from the saved JSON)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..train.regression import predict, x_regression


def show(target_fun, params, save=""):
    """Target and prediction of the trained circuit."""
    x = np.asarray(x_regression)
    fig, ax = plt.subplots()
    ax.plot(x, target_fun(x), label="Target", color="orange")
    ax.plot(x, predict(params), label="Predicted", color="blue")
    ax.set(xlim=(-1.05, 1.05), ylim=(-1.1, 1.1), xlabel="x", ylabel=r"$\langle Z\rangle$")
    ax.legend(loc="upper left")
    if save:
        fig.savefig(save)
    return fig


def animated(target_fun, history, save="", max_frames=None):
    """Animation of the prediction over training, from `history` of regression.train."""
    from IPython.display import HTML

    x = np.asarray(x_regression)
    fig, ax = plt.subplots()
    ax.set(xlim=(-1.05, 1.05), ylim=(-1.1, 1.1), xlabel="x", ylabel=r"$\langle Z\rangle$")
    ax.plot(x, target_fun(x), label="Target", color="orange")
    (line,) = ax.plot([], [], label="Predicted", color="blue")
    text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper right")
    n = len(history["f"])
    frames = np.linspace(0, n - 1, max_frames, dtype=int) if max_frames else range(n)

    def update(k):
        line.set_data(x, history["f"][k])
        text.set_text(f"Epoch {history['epoch'][k]}, MSE {history['loss'][k]:.2e}")
        return line, text

    anim = FuncAnimation(fig, update, frames=frames, blit=True, interval=100)
    if save:
        anim.save(save, writer="pillow" if str(save).endswith(".gif") else "ffmpeg")
    return HTML(anim.to_jshtml())
