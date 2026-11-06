"""Maps between the data and the range [-1, 1] of the expectation values <Z>."""
import numpy as np
from scipy.special import erf, erfinv


class Expectations:
    """Standardise every feature, then erf(z / scale): heavy tails are squeezed into (-1, 1).
    Used for the CLIC calorimeter energies."""

    def __init__(self, scale: float = 4.0):
        self.scale = scale

    def fit(self, x):
        self.mean, self.std = np.mean(x, 0), np.std(x, 0)
        return self

    def encode(self, x):
        return erf((x - self.mean) / self.std / self.scale)

    def decode(self, a):
        a = np.clip(a, -1 + 1e-12, 1 - 1e-12)
        return self.scale * erfinv(a) * self.std + self.mean

    def state(self) -> dict:
        return {"type": "Expectations", "scale": self.scale,
                "mean": np.asarray(self.mean).tolist(), "std": np.asarray(self.std).tolist()}


class Linear:
    """Affine map [lo, hi] -> [-1, 1]; for MNIST grey levels in [0, 1]: <Z> = 2p - 1."""

    def __init__(self, lo: float = 0.0, hi: float = 1.0):
        self.lo, self.hi = lo, hi

    def fit(self, x):
        return self

    def encode(self, x):
        return 2 * (np.asarray(x) - self.lo) / (self.hi - self.lo) - 1

    def decode(self, a):
        return self.lo + (np.clip(a, -1, 1) + 1) / 2 * (self.hi - self.lo)

    def state(self) -> dict:
        return {"type": "Linear", "lo": self.lo, "hi": self.hi}
