"""Regression of four 1D functions with the single-qubit universal approximator.

Writes results/fits/fit_<name>.json: x grid, target, and at every epoch the MSE, the
prediction f(x) = <Z>(x) and the parameters phi, w, alpha (epoch 0 = initialisation).

    python scripts/fit.py [name ...]
"""
import argparse

import _common  # noqa: F401  (sets sys.path)
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from mua.train import regression  # noqa: E402
from mua.utils import save_json, to_list  # noqa: E402

FITS = {
    # name: (latex, function, depth, epochs, lr) -- as in notebooks/fit.ipynb
    "tanh": (r"\tanh(5x)", lambda x: np.tanh(5 * x), 5, 150, 0.1),
    "upsin": (r"\big(x + \sin(3\pi x)\big)/2", lambda x: (x + np.sin(3 * np.pi * x)) / 2, 15, 150, 0.1),
    "squarewave": (r"\mathrm{sign}\big(\cos(2\pi x)\big)", lambda x: np.sign(np.cos(2 * np.pi * x)),
                   100, 600, 0.3),
    "veryhard": (r"\mathrm{sign}\big(\cos(2\pi x)\big)|x|",
                 lambda x: np.sign(np.cos(2 * np.pi * x)) * np.abs(x), 100, 2000, 0.01),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(FITS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for name in args.names:
        latex, fun, depth, epochs, lr = FITS[name]
        params, h = regression.train(fun, depth, epochs, lr, seed=args.seed, progress_bar=False)
        x = np.asarray(regression.x_regression)
        print(f"{name:10s} depth {depth:3d}: MSE {h['loss'][0]:.2e} -> {h['loss'][-1]:.2e}")
        save_json({
            "name": name, "target_latex": latex, "depth": depth, "num_epochs": epochs, "lr": lr,
            "optimizer": "adam", "loss": "MSE", "seed": args.seed,
            "init": "phi, w, alpha ~ U[0,1)",
            "circuit": "prod_i RY(2 phi_i) RZ(2 (w_i x + alpha_i)) |0>, output <Z>",
            "note": "history[k] after k full-batch Adam steps; k = 0 is the initialisation",
            "x": to_list(x), "target": to_list(fun(x)),
            "history": {"epoch": h["epoch"], "loss": h["loss"],
                        **{k: to_list(h[k]) for k in ("f", "phi", "w", "alpha")}},
            "provenance": _common.provenance(),
        }, _common.RESULTS / "fits" / f"fit_{name}.json")


if __name__ == "__main__":
    main()
