"""Generative training of the multi universal approximator with an MMD loss.

One qubit per feature; the noise z ~ U[0, 1)^depth is shared by all qubits and re-uploaded
in every layer (w_{q,i} . z + alpha_{q,i}), so the noise dimension equals the depth."""
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from ..losses import corr, mmd
from ..unitaries.multiUuat import batch_multi_Uuat
from ..utils import median_dist


def random_noise(num_vecs: int, depth: int, rng: np.random.Generator) -> jnp.ndarray:
    """Uniform noise in [0, 1), shape (num_vecs, depth)."""
    return jnp.array(rng.random((num_vecs, depth)))


def init_params(num_qubits: int, depth: int, w_scale: float = 1.0,
                rng: np.random.Generator | None = None) -> dict:
    """phi, alpha ~ U[0, 1); w ~ w_scale * U[0, 1).

    With w ~ U[0, 1) the re-uploading angle w . z grows like depth / 4: for deep circuits the
    output oscillates quickly with z and the MMD gradients average out (a barren plateau in
    the noise); w_scale ~ 1 / depth keeps the angles O(1)."""
    rng = rng or np.random.default_rng()
    return {"phi": jnp.array(rng.random((num_qubits, depth))),
            "w": jnp.array(w_scale * rng.random((num_qubits, depth, depth))),
            "alpha": jnp.array(rng.random((num_qubits, depth)))}


_generate = jax.jit(lambda p, z: batch_multi_Uuat(z, p["phi"], p["w"], p["alpha"]))


def generate(params: dict, num: int, rng: np.random.Generator, chunk: int = 2000) -> np.ndarray:
    """`num` samples of <Z>, shape (num, num_qubits)."""
    depth = params["phi"].shape[1]
    out = [np.asarray(_generate(params, random_noise(min(chunk, num - i), depth, rng)))
           for i in range(0, num, chunk)]
    return np.concatenate(out)


def train(data: np.ndarray, depth: int, num_epochs: int = 20, batch_size: int = 1000,
          lr: float = 1e-2, bandwidth_factors=(0.25, 0.5, 1.0, 2.0, 4.0),
          bandwidths=None, w_scale: float = 1.0, test_data: np.ndarray | None = None,
          eval_size: int = 1000, seed: int | None = None, progress_bar: bool = True,
          schedule: str = "constant", max_time: float | None = None, callback=None,
          callback_every: int = 50):
    """
    Parameters
    ----------
    data : (N, Q) training vectors in [-1, 1] (already encoded).
    depth : layers per qubit (= noise dimension).
    num_epochs : passes over the training set; one Adam step per batch of `batch_size`
        (the last incomplete batch is dropped, to keep shapes static for jit).
    bandwidths : kernel widths h of the MMD; default `bandwidth_factors` x the median
        pairwise distance of the training data.
    test_data : (M, Q) held-out vectors; MMD and correlation distance are evaluated on
        `eval_size` of them with fixed noise after every epoch (default: training data).
    schedule : "constant" or "cosine" (cosine decay from `lr` to `lr` / 100 over all steps).
    max_time : stop after the first epoch that ends past `max_time` seconds (wall clock).
    callback : callback(params, history), called every `callback_every` epochs (checkpoints).

    The training noise and shuffling use jax.random (key from `seed`), the initialisation and
    the evaluation noise numpy (`seed`).

    Returns
    -------
    params : dict phi (Q, D), w (Q, D, D), alpha (Q, D)
    history : dict
        step_loss (MMD of every training batch), epoch, mmd, corr (held-out, every epoch,
        epoch 0 = initialisation), bandwidths, time
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    num_samples, num_qubits = data.shape
    num_batches = num_samples // batch_size

    if bandwidths is None:
        h = median_dist(data, 1000, seed=seed)
        bandwidths = [k * h for k in bandwidth_factors]
    bandwidths = [float(b) for b in bandwidths]

    params = init_params(num_qubits, depth, w_scale, rng)
    if schedule == "cosine":
        lr = optax.cosine_decay_schedule(lr, num_epochs * num_batches, alpha=0.01)
    elif schedule != "constant":
        raise ValueError(f"unknown schedule {schedule!r}")
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(p, z, y):
        return mmd(batch_multi_Uuat(z, p["phi"], p["w"], p["alpha"]), y, bandwidths)

    # one compiled call per epoch: shuffle, draw the noise and run all Adam steps on the
    # device (lax.scan over batches); the data is an argument, not a baked-in constant
    @jax.jit
    def run_epoch(p, s, key, x):
        k_perm, k_noise = jax.random.split(key)
        idx = jax.random.permutation(k_perm, num_samples)[:num_batches * batch_size]
        idx = idx.reshape(num_batches, batch_size)
        zs = jax.random.uniform(k_noise, (num_batches, batch_size, depth), x.dtype)

        def step(carry, a):
            p, s = carry
            i, z = a
            loss, g = jax.value_and_grad(loss_fn)(p, z, x[i])
            u, s = optimizer.update(g, s, p)
            return (optax.apply_updates(p, u), s), loss

        (p, s), losses = jax.lax.scan(step, (p, s), (idx, zs))
        return p, s, losses

    x_dev = jnp.asarray(data)
    key = jax.random.PRNGKey(0 if seed is None else seed)

    ref = np.asarray(test_data if test_data is not None else data)
    ref = ref[rng.choice(len(ref), min(eval_size, len(ref)), replace=False)]
    keep = np.flatnonzero(ref.std(0) > 0)
    z_eval = random_noise(len(ref), depth, rng)
    ref_dev = jnp.asarray(ref)
    mmd_eval = jax.jit(lambda a, r: mmd(a, r, bandwidths))

    history = {"step_loss": [], "epoch": [], "mmd": [], "corr": [], "time": [],
               "bandwidths": bandwidths}
    t0 = time.time()

    def log(epoch, p):
        fake = _generate(p, z_eval)
        history["epoch"].append(epoch)
        history["mmd"].append(float(mmd_eval(fake, ref_dev)))
        history["corr"].append(corr(np.asarray(fake), ref, keep))
        history["time"].append(time.time() - t0)

    log(0, params)
    bar = tqdm.trange(1, num_epochs + 1, disable=not progress_bar, ncols=100)
    for epoch in bar:
        key, k = jax.random.split(key)
        params, opt_state, losses = run_epoch(params, opt_state, k, x_dev)
        history["step_loss"].extend(np.asarray(losses).tolist())
        log(epoch, params)
        bar.set_description(f"MMD {history['mmd'][-1]:.2e} corr {history['corr'][-1]:.2e}")
        if not progress_bar:
            print(f"epoch {epoch}/{num_epochs}: MMD {history['mmd'][-1]:.3e}, "
                  f"corr {history['corr'][-1]:.3e} ({history['time'][-1]:.0f}s)", flush=True)
        stop = max_time is not None and history["time"][-1] > max_time
        if callback is not None and (epoch % callback_every == 0 or stop):
            callback(params, history)
        if stop:
            print(f"time budget reached after epoch {epoch}", flush=True)
            break
    return params, history
