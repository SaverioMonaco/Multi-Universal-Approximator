from typing import Sequence

import jax.numpy as jnp

from ..utils import gaussian_kernel, sq_cdist


def mmd(x: jnp.ndarray, y: jnp.ndarray, bandwidths: Sequence[float]) -> jnp.ndarray:
    """Biased MMD^2 estimate with a Gaussian kernel, averaged over several bandwidths.

    Parameters
    ----------
    x, y : (n, d), (m, d) samples.
    bandwidths : kernel widths h (same units as the data), k(a, b) = exp(-|a-b|^2 / (2 h^2)).
    """
    XX, YY, XY = sq_cdist(x, x), sq_cdist(y, y), sq_cdist(x, y)
    return jnp.mean(jnp.stack([
        gaussian_kernel(XX, h).mean() + gaussian_kernel(YY, h).mean()
        - 2 * gaussian_kernel(XY, h).mean()
        for h in bandwidths
    ]))
