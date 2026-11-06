"""Regression of a 1D function f: [-1, 1] -> [-1, 1] with the single-qubit approximator."""
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from ..losses.mse import mse
from ..unitaries.Uuat import Uuat

x_regression = jnp.linspace(-1, 1, 100)

# batch over scalar x: (B,), (D,), (D,), (D,) -> (B,)
jv_Uuat = jax.jit(jax.vmap(Uuat, in_axes=(0, None, None, None)))


def init_params(depth: int, seed: int | None = None) -> dict:
    """phi, w, alpha ~ U[0, 1) (as in v1)."""
    rng = np.random.default_rng(seed)
    return {"phi": jnp.array(rng.random(depth)), "w": jnp.array(rng.random(depth)),
            "alpha": jnp.array(rng.random(depth))}


def predict(params: dict, x=x_regression) -> jnp.ndarray:
    return jv_Uuat(x, params["phi"], params["w"], params["alpha"])


def train(target_fun: Callable, depth: int, num_epochs: int, lr: float = 0.1,
          seed: int | None = None, x=x_regression, record: bool = True,
          progress_bar: bool = True) -> tuple[dict, dict]:
    """Full-batch Adam on the MSE between f(x) and <Z>(x).

    Returns
    -------
    params : dict with phi, w, alpha (each shape (depth,))
    history : dict of lists, entry k after k Adam steps (k = 0: initialisation):
        epoch, loss, and if `record` also f (prediction on x), phi, w, alpha
    """
    x = jnp.asarray(x)
    y = jnp.asarray(target_fun(np.asarray(x)))
    assert (y >= -1).all() and (y <= 1).all(), "target must be in [-1, 1]"

    params = init_params(depth, seed)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return mse(predict(p, x), y)

    @jax.jit
    def step(p, s):
        loss, g = jax.value_and_grad(loss_fn)(p)
        u, s = optimizer.update(g, s)
        return optax.apply_updates(p, u), s, loss

    history = {"epoch": [], "loss": []}
    if record:
        history.update({"f": [], "phi": [], "w": [], "alpha": []})

    def log(epoch, p):
        f = predict(p, x)
        history["epoch"].append(epoch)
        history["loss"].append(float(mse(f, y)))
        if record:
            history["f"].append(np.asarray(f))
            for k in ("phi", "w", "alpha"):
                history[k].append(np.asarray(p[k]))

    log(0, params)
    bar = tqdm.trange(1, num_epochs + 1, disable=not progress_bar, ncols=90)
    for epoch in bar:
        params, opt_state, _ = step(params, opt_state)
        log(epoch, params)
        bar.set_description(f"MSE {history['loss'][-1]:.2e}")
    return params, history
