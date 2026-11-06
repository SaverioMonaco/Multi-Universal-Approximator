"""Distances, kernels and small helpers shared by losses, metrics and training."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def sq_cdist(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Pairwise squared Euclidean distances, (n, d), (m, d) -> (n, m).

    Uses |x|^2 + |y|^2 - 2 x.y, so memory is O(n m) and not O(n m d)."""
    d2 = jnp.sum(x**2, 1)[:, None] + jnp.sum(y**2, 1)[None, :] - 2 * x @ y.T
    return jnp.maximum(d2, 0.0)


def cdist(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Pairwise Euclidean distances."""
    return jnp.sqrt(sq_cdist(x, y))


def gaussian_kernel(D2: jnp.ndarray, h: float) -> jnp.ndarray:
    """Gaussian kernel exp(-d^2 / (2 h^2)) on squared distances, bandwidth h (distance units)."""
    return jnp.exp(-D2 / (2 * h**2))


def median_dist(x: np.ndarray, subset_size: int = 1000, seed: int | None = 0) -> float:
    """Median pairwise Euclidean distance on a random subset (the 'median heuristic')."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    sub = x[rng.choice(len(x), min(subset_size, len(x)), replace=False)]
    d2 = np.asarray(sq_cdist(jnp.array(sub), jnp.array(sub)))
    return float(np.sqrt(np.median(d2[np.triu_indices(len(sub), 1)])))


def to_list(a, digits: int = 6):
    """Rounded nested list (compact JSON)."""
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def save_json(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)
    return path


COLORS = {"mse": "dodgerblue", "corr": "orange", "mmd": "violet"}
