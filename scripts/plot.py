"""Figures from the results JSON (scripts/fit.py, scripts/gen.py), as in the thesis.

  fit_<name>.pdf               regression of tanh, upsin, square wave, |x| square wave: target,
                               prediction at two intermediate epochs and at the end, MSE below
  fit_<name>.gif               the same, animated over training (skip with --no-anim)
  spectrum_<name>.pdf          Fourier amplitude spectrum of the target and the final fit
  loss_<clic|mnist>_*.pdf      held-out MMD^2 and correlation distance during training
  samples_clic_*.pdf           CLIC: mean pixel energies, energy sum, correlation matrices
  samples_mnist_*.pdf          MNIST: test images vs generated images
  stats_mnist_*.pdf            MNIST: pixel mean / std images, grey-level distribution

    python scripts/plot.py [NAME ...] [--data DIR] [--out DIR] [--no-tex] [--no-anim]

NAME restricts the figures to those fits (squarewave, ...) and/or generation runs
(mnist_d64_xlong, clic_q8_d8, ...); without names everything is plotted. By default reads results/{fits,clic,mnist}/*.json and writes results/plots/. The thesis
style uses mplhep (optional) and a LaTeX install (disable with --no-tex).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.special import erf

import _common


def setup_style(usetex: bool = True) -> None:
    """Shared thesis style (PhD/tex/common/style.py)."""
    if usetex:
        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["font.family"] = "serif"
    try:
        import mplhep as hep
        plt.style.use(hep.style.ROOT)
    except ImportError:
        pass
    plt.rcParams.update({
        "legend.edgecolor": "black", "legend.frameon": True, "legend.fancybox": False,
        "legend.framealpha": 0.8, "legend.borderpad": 0.4, "legend.borderaxespad": 0.5,
        "legend.handlelength": 1.0,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.25, "ytick.minor.width": 0.25,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 3, "ytick.minor.size": 3,
        "axes.linewidth": 1.2,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.titlesize": 10,
    })


# thesis palette (common/tikzhelpers.tex)
MAINCOLOR = "#00A6EB"  # desyblue
SECONDARYCOLOR = "#F38E00"  # desyorange
TERTIARYCOLOR = "#777777"  # desygray
DARKCOLOR = "#00557a"  # dark shade of desyblue (final prediction)
GRID = "#e5e4df"
SNAP_CMAP = LinearSegmentedColormap.from_list("snap", ["#bfe6f8", MAINCOLOR, DARKCOLOR])

def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out / name}")


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")


def style_legend(legend) -> None:
    frame = legend.get_frame()
    frame.set_linewidth(0.5)
    frame.set_alpha(None)
    frame.set_facecolor((1, 1, 1, 0.85))
    frame.set_edgecolor((0, 0, 0, 0.3))


def thin_thousands(n: int) -> str:
    return f"{n:,}".replace(",", r"\,")


# %%
# --- regression -----------------------------------------------------------------------
FITS = ["tanh", "upsin", "squarewave", "veryhard"]  # -> fit_<name>.pdf


def snapshot_epochs(loss: np.ndarray, k: int = 2) -> list[int]:
    """k intermediate epochs, where the running-best MSE first crosses k levels evenly
    spaced in log between the initial and the final MSE."""
    best = np.minimum.accumulate(loss)
    levels = np.geomspace(loss[0], loss[-1], k + 2)[1:-1]
    return sorted({int(np.argmax(best <= lv)) for lv in levels})


def plot_fit(d: dict) -> plt.Figure:
    h = d["history"]
    x, target, f = np.array(d["x"]), np.array(d["target"]), np.array(h["f"])
    epochs, loss = np.array(h["epoch"]), np.array(h["loss"])
    last = int(epochs[-1])
    snaps = snapshot_epochs(loss)
    colors = [SNAP_CMAP(t) for t in np.linspace(0.15, 0.55, len(snaps))]

    fig, (ax, axl) = plt.subplots(2, 1, figsize=(3.4, 3.8),
                                  gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.5})
    ax.plot(x, target, color=SECONDARYCOLOR, linewidth=2.6, alpha=0.55, zorder=2,
            label="Target $f(x)$")
    for e, c in zip(snaps, colors):
        ax.plot(x, f[e], color=c, linewidth=1.0, linestyle="--", zorder=3, label=f"Epoch {e}")
    ax.plot(x, f[last], color=DARKCOLOR, linewidth=1.4, zorder=4, label=f"Epoch {last}")
    ax.plot([], [], " ", label=f"($N = {d['depth']}$ layers)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\langle Z \rangle$")
    style_axes(ax)
    style_legend(ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand",
                           ncol=2, fontsize=7, handlelength=1.6, borderpad=0.3,
                           labelspacing=0.25, columnspacing=0.8, borderaxespad=0))

    axl.plot(epochs[1:], loss[1:], color=TERTIARYCOLOR, linewidth=1.0)
    for e, c in zip(snaps + [last], colors + [DARKCOLOR]):
        axl.plot(e, loss[e], "o", color=c, markersize=3.5, zorder=3)
    axl.set_xscale("log")
    axl.set_yscale("log")
    axl.set_xlim(1, last)
    axl.set_xlabel("Epoch")
    axl.set_ylabel("MSE")
    style_axes(axl)
    return fig


def plot_spectrum(d: dict) -> plt.Figure:
    """Fourier amplitude spectrum of the target (bars) and of the final prediction."""
    x, target = np.array(d["x"]), np.array(d["target"])
    last = int(d["history"]["epoch"][-1])
    freq, amp_t = spectrum(x, target)
    _, amp_f = spectrum(x, np.array(d["history"]["f"][last]))
    floor = 1e-3 * max(amp_t.max(), amp_f.max())  # log axis: clip exact zeros
    width = 0.8 * (freq[1] - freq[0])

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.bar(freq, np.maximum(amp_t, floor) - floor, width=width, bottom=floor,
           color=SECONDARYCOLOR, alpha=0.55, zorder=2, label="Target $f(x)$")
    ax.plot(freq, np.maximum(amp_f, floor), color=DARKCOLOR, linewidth=0.9, marker="o",
            markersize=1.8, zorder=3, label=f"Epoch {last}")
    ax.plot([], [], " ", label=f"($N = {d['depth']}$ layers)")
    ax.set_yscale("log")
    ax.set_xlim(-width, freq[-1] + width)
    ax.set_ylim(floor, None)
    ax.set_xlabel(r"Frequency $\nu$ (cycles per unit $x$)")
    ax.set_ylabel(r"$|\hat f(\nu)|$")
    style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()  # bars come last; put the target first
    order = [len(handles) - 1, *range(len(handles) - 1)]
    style_legend(ax.legend([handles[i] for i in order], [labels[i] for i in order],
                           loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand",
                           ncol=3, fontsize=7, handlelength=1.2, borderpad=0.3,
                           columnspacing=0.8, borderaxespad=0))
    return fig


def spectrum(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Fourier amplitude spectrum of y sampled on the uniform grid x: a cosine
    of amplitude A at frequency nu gives a bar of height A at nu (the grid is treated as
    one period, so a function with f(-1) != f(1) picks up a jump at the edges)."""
    amp = np.abs(np.fft.rfft(y)) / len(y)
    amp[1:] *= 2
    return np.fft.rfftfreq(len(y), d=x[1] - x[0]), amp


def animate_fit(d: dict, frames: int = 120, fps: int = 20, hold: float = 1.5) -> FuncAnimation:
    """Prediction over training: f(x) at log-spaced epochs (most of the change is early),
    the MSE curve below drawn up to the current epoch; the last frame is held `hold` s."""
    h = d["history"]
    x, target, f = np.array(d["x"]), np.array(d["target"]), np.array(h["f"])
    epochs, loss = np.array(h["epoch"]), np.array(h["loss"])
    last = int(epochs[-1])
    ks = np.unique(np.r_[0, np.geomspace(1, last, frames - 1).round().astype(int)])
    ks = np.r_[ks, [last] * int(hold * fps)]

    fig, (ax, axl) = plt.subplots(2, 1, figsize=(3.4, 3.8),
                                  gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.5})
    ax.plot(x, target, color=SECONDARYCOLOR, linewidth=2.6, alpha=0.55, zorder=2,
            label="Target $f(x)$")
    (line,) = ax.plot(x, f[0], color=DARKCOLOR, linewidth=1.4, zorder=4, label="Prediction")
    ax.plot([], [], " ", label=f"($N = {d['depth']}$ layers)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\langle Z \rangle$")
    style_axes(ax)
    style_legend(ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand",
                           ncol=2, fontsize=7, handlelength=1.6, borderpad=0.3,
                           labelspacing=0.25, columnspacing=0.8, borderaxespad=0))
    text = ax.text(0.03, 0.04, "", transform=ax.transAxes, fontsize=7)

    axl.plot(epochs[1:], loss[1:], color=TERTIARYCOLOR, linewidth=1.0, alpha=0.25)
    (trail,) = axl.plot([], [], color=TERTIARYCOLOR, linewidth=1.0)
    (dot,) = axl.plot([], [], "o", color=DARKCOLOR, markersize=3.5, zorder=3)
    axl.set_xscale("log")
    axl.set_yscale("log")
    axl.set_xlim(1, last)
    axl.set_xlabel("Epoch")
    axl.set_ylabel("MSE")
    style_axes(axl)

    def update(k):
        line.set_ydata(f[k])
        trail.set_data(epochs[1:k + 1], loss[1:k + 1])
        dot.set_data([max(k, 1)], [loss[k]])
        text.set_text(f"Epoch {k}, MSE {loss[k]:.1e}")
        return line, trail, dot, text

    return FuncAnimation(fig, update, frames=ks, interval=1000 / fps, blit=True)


def save_anim(anim: FuncAnimation, out: Path, name: str, fps: int = 20, dpi: int = 150) -> None:
    out.mkdir(parents=True, exist_ok=True)
    anim.save(out / name, writer=PillowWriter(fps=fps), dpi=dpi,
              savefig_kwargs={"facecolor": "white"})
    plt.close(anim._fig)
    print(f"saved {out / name}")


# %%
# --- generation: shared ---------------------------------------------------------------
def encode(d: dict, x: np.ndarray) -> np.ndarray:
    """The encoding used in training (mua.scalers), from the stored scaler state."""
    s = d["scaler"]
    if s["type"] == "Expectations":
        return erf((x - np.array(s["mean"])) / np.array(s["std"]) / s["scale"])
    return 2 * (x - s["lo"]) / (s["hi"] - s["lo"]) - 1


def mmd2(x: np.ndarray, y: np.ndarray, bandwidths) -> float:
    """Same estimator as the training loss (mua.losses.mmd)."""
    def sq(a, b):
        return np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T, 0)
    xx, yy, xy = sq(x, x), sq(y, y), sq(x, y)
    return float(np.mean([np.exp(-xx / (2 * h**2)).mean() + np.exp(-yy / (2 * h**2)).mean()
                          - 2 * np.exp(-xy / (2 * h**2)).mean() for h in bandwidths]))


def corr_dist(x: np.ndarray, y: np.ndarray, keep) -> float:
    """Same metric as mua.losses.corr (constant pixels have correlation 0)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.nan_to_num(np.corrcoef(x[:, keep], rowvar=False))
        cy = np.nan_to_num(np.corrcoef(y[:, keep], rowvar=False))
    return float(np.mean((cx - cy) ** 2))


def floors(d: dict) -> tuple[float, float]:
    """The metrics between two disjoint sets of test samples: what a perfect generator
    reaches with this number of samples (eval_size each, or half of the stored ones)."""
    real = encode(d, np.array(d["samples"]["real"]))
    n = min(d["eval_size"], len(real) // 2)
    a, b = real[:n], real[n:2 * n]
    keep = np.flatnonzero(real.std(0) > 0)
    return mmd2(a, b, d["bandwidths"]), corr_dist(a, b, keep)


def smooth(y, w: int = 9):
    """Centred running mean (edges use the available window)."""
    y = np.asarray(y, float)
    return np.convolve(y, np.ones(w), "same") / np.convolve(np.ones_like(y), np.ones(w), "same")


def plot_loss(d: dict) -> plt.Figure:
    h = d["history"]
    ep, mmd_te, corr = np.array(h["epoch"]), np.array(h["mmd"]), np.array(h["corr"])
    step = np.array(h["step_loss"])
    step_ep = np.arange(1, len(step) + 1) * ep[-1] / len(step)  # step -> fractional epoch
    mmd_floor, corr_floor = floors(d)
    what = "test images" if d["dataset"].startswith("MNIST") else "test showers"

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(4.2, 3.6), sharex=True,
                                 gridspec_kw={"hspace": 0.12})
    a1.plot(step_ep, np.clip(step, 1e-8, None), color=MAINCOLOR, alpha=0.2, linewidth=0.5,
            label="Training batches")
    a1.plot(ep, np.clip(mmd_te, 1e-8, None), color=MAINCOLOR, linewidth=1.4, marker="o",
            markersize=2, label=f"Held-out {what}")
    a2.plot(ep, corr, color=SECONDARYCOLOR, linewidth=1.4, marker="o", markersize=2)
    for ax, floor, lab in ((a1, mmd_floor, r"$\mathrm{MMD}^2$"), (a2, corr_floor, "Corr. distance")):
        ax.axhline(floor, color=TERTIARYCOLOR, linestyle=":", linewidth=1.2,
                   label="Data vs data (sample floor)")
        ax.set_yscale("log")
        ax.set_ylabel(lab)
        style_axes(ax)
    style_legend(a1.legend(loc="upper right", fontsize=7))
    a2.set_xlabel("Epoch")
    a2.set_xlim(0, ep[-1])
    name = "MNIST" if d["dataset"].startswith("MNIST") else "CLIC"
    a1.set_title(f"{name}, {d['num_qubits']} qubits, depth $N = {d['depth']}$, "
                 f"{thin_thousands(d['num_params'])} parameters", fontsize=9)
    return fig


# %%
# --- CLIC -----------------------------------------------------------------------------
def plot_samples_clic(d: dict) -> plt.Figure:
    real, fake = np.array(d["samples"]["real"]), np.array(d["samples"]["fake"])
    npx = d["num_qubits"]
    px = np.arange(1, npx + 1)

    fig = plt.figure(figsize=(9.0, 2.7))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1.1, 1, 1], wspace=0.45)
    ax_p, ax_e = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax_cr, ax_cf = fig.add_subplot(gs[2]), fig.add_subplot(gs[3])

    # mean shower profile with the 16-84 % band
    for e, c, lab, ls in ((real, SECONDARYCOLOR, "CLIC (test)", "-"), (fake, MAINCOLOR, "MUA", "--")):
        lo, hi = np.percentile(e, [16, 84], axis=0)
        ax_p.fill_between(px, lo, hi, color=c, alpha=0.18, linewidth=0, step="mid")
        ax_p.step(px, e.mean(0), where="mid", color=c, linestyle=ls, linewidth=1.3, label=lab)
    ax_p.set_xlabel("Pixel")
    ax_p.set_ylabel("Energy")
    ax_p.set_xlim(0.5, npx + 0.5)
    ax_p.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    style_axes(ax_p)
    style_legend(ax_p.legend(loc="upper right", fontsize=7))

    # energy sum
    er, ef = real.sum(1), fake.sum(1)
    bins = np.linspace(min(er.min(), ef.min()),
                       max(np.percentile(er, 99.9), np.percentile(ef, 99.9)), 40)
    ax_e.hist(er, bins=bins, density=True, histtype="stepfilled", color=SECONDARYCOLOR, alpha=0.3)
    ax_e.hist(er, bins=bins, density=True, histtype="step", color=SECONDARYCOLOR, linewidth=1.0)
    ax_e.hist(ef, bins=bins, density=True, histtype="step", color=MAINCOLOR, linewidth=1.3,
              linestyle="--")
    ax_e.set_xlabel(r"$\sum_i E_i$")
    ax_e.set_ylabel("Density")
    style_axes(ax_e)

    # correlation matrices
    for ax, e, t in ((ax_cr, real, "CLIC (test)"), (ax_cf, fake, "MUA")):
        im = ax.imshow(np.corrcoef(e, rowvar=False), vmin=-1, vmax=1, cmap="RdBu_r",
                       extent=(0.5, npx + 0.5, npx + 0.5, 0.5))
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("Pixel")
        ax.tick_params(which="both", top=False, right=False)
        ax.minorticks_off()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax_cr.set_ylabel("Pixel")
    ax_cf.set_yticklabels([])
    cb = fig.colorbar(im, ax=[ax_cr, ax_cf], fraction=0.03, pad=0.03)
    cb.set_label("Pearson correlation")
    cb.ax.minorticks_off()
    return fig


# %%
# --- MNIST ----------------------------------------------------------------------------
def tile(imgs: np.ndarray, rows: int, cols: int, shape=(28, 28), pad: int = 2) -> np.ndarray:
    """Grid of images, ink dark on white."""
    h, w = shape
    out = np.ones((rows * (h + pad) + pad, cols * (w + pad) + pad))
    for k in range(rows * cols):
        r, c = divmod(k, cols)
        out[pad + r * (h + pad):pad + r * (h + pad) + h,
            pad + c * (w + pad):pad + c * (w + pad) + w] = 1 - imgs[k].reshape(shape)
    return out


def plot_samples_mnist(d: dict, rows: int = 6, cols: int = 10) -> plt.Figure:
    real, fake = np.array(d["samples"]["real"]), np.array(d["samples"]["fake"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.9), gridspec_kw={"wspace": 0.06})
    for ax, imgs, title in ((axes[0], real, "MNIST (test)"),
                            (axes[1], fake, f"Multi universal approximator, $N = {d['depth']}$")):
        ax.imshow(tile(imgs, rows, cols, d["shape"]), cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    return fig


def plot_stats_mnist(d: dict) -> plt.Figure:
    real, fake = np.array(d["samples"]["real"]), np.array(d["samples"]["fake"])
    shape = d["shape"]
    fig = plt.figure(figsize=(9.0, 2.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.6], wspace=0.15)
    panels = [(real.mean(0), "Mean, MNIST"), (fake.mean(0), "Mean, MUA"),
              (real.std(0), "Std., MNIST"), (fake.std(0), "Std., MUA")]
    for k, (img, title) in enumerate(panels):
        ax = fig.add_subplot(gs[k])
        ax.imshow(img.reshape(shape), cmap="Blues", vmin=0, vmax=1 if k < 2 else 0.5,
                  interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    ax = fig.add_subplot(gs[4])
    bins = np.linspace(0, 1, 41)
    ax.hist(real.ravel(), bins=bins, density=True, histtype="stepfilled", color=SECONDARYCOLOR,
            alpha=0.3, label="MNIST (test)")
    ax.hist(real.ravel(), bins=bins, density=True, histtype="step", color=SECONDARYCOLOR, linewidth=1.0)
    ax.hist(fake.ravel(), bins=bins, density=True, histtype="step", color=MAINCOLOR, linewidth=1.3,
            linestyle="--", label="MUA")
    ax.set_yscale("log")
    ax.set_xlabel("Grey level $p$")
    ax.set_ylabel("Density")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    style_axes(ax)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(right=True, left=False, which="both")
    style_legend(ax.legend(loc="center", fontsize=7))
    return fig


# %%
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*",
                    help="only these fits / generation runs (default: all)")
    ap.add_argument("--data", type=Path, default=_common.RESULTS,
                    help="results folder: fits/fit_*.json, gen_*.json anywhere below")
    ap.add_argument("--out", type=Path, default=_common.RESULTS / "plots")
    ap.add_argument("--no-tex", action="store_true", help="do not use LaTeX for text")
    ap.add_argument("--no-anim", action="store_true", help="skip the fit_<name>.gif animations")
    args = ap.parse_args()
    setup_style(usetex=not args.no_tex)

    fits = args.data / "fits" if (args.data / "fits").is_dir() else args.data
    gens = {p.stem.removeprefix("gen_"): p for p in sorted(args.data.rglob("gen_*.json"))}
    unknown = set(args.names) - set(FITS) - set(gens)
    if unknown:
        ap.error(f"unknown name(s) {', '.join(sorted(unknown))}; "
                 f"fits: {', '.join(FITS)}; runs: {', '.join(gens) or 'none found'}")
    wanted = lambda name: not args.names or name in args.names

    for name in FITS:
        p = fits / f"fit_{name}.json"
        if wanted(name) and p.exists():
            d = load_json(p)
            save(plot_fit(d), args.out, f"fit_{name}.pdf")
            save(plot_spectrum(d), args.out, f"spectrum_{name}.pdf")
            if not args.no_anim:
                save_anim(animate_fit(d), args.out, f"fit_{name}.gif")
    for name, p in gens.items():
        if not wanted(name):
            continue
        d = load_json(p)
        save(plot_loss(d), args.out, f"loss_{name}.pdf")
        if d["dataset"].startswith("MNIST"):
            save(plot_samples_mnist(d), args.out, f"samples_{name}.pdf")
            save(plot_stats_mnist(d), args.out, f"stats_{name}.pdf")
        else:
            save(plot_samples_clic(d), args.out, f"samples_{name}.pdf")


if __name__ == "__main__":
    main()
