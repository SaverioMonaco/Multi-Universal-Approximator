"""The pure-JAX state vector agrees with the PennyLane QNode (run: python -m pytest tests)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mua.unitaries import Uuat as U  # noqa: E402


def test_scalar_input():
    rng = np.random.default_rng(0)
    d = 7
    args = (0.3, rng.random(d), rng.random(d), rng.random(d))
    assert np.isclose(U.Uuat(*args), U.Uuat_qml(*args), atol=1e-6)


def test_vector_input():
    rng = np.random.default_rng(1)
    d = 5
    args = (rng.random(d), rng.random(d), rng.random((d, d)), rng.random(d))
    assert np.isclose(U.Uuat(*args), U.Uuat_qml(*args), atol=1e-6)


if __name__ == "__main__":
    test_scalar_input(); test_vector_input(); print("backend OK")
