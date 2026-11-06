"""The pure-JAX state vector agrees with the PennyLane QNode (run: python -m pytest tests)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mua.unitaries import Uuat as U  # noqa: E402
from mua.unitaries.multiUuat import batch_multi_Uuat, batch_multi_Uuat_sv  # noqa: E402


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


def test_bloch_batch():
    """The Bloch-vector batch forward (used in training) equals the vmapped state vector."""
    rng = np.random.default_rng(2)
    B, Q, D = 6, 4, 5
    args = (rng.random((B, D)), rng.random((Q, D)), rng.random((Q, D, D)), rng.random((Q, D)))
    assert np.allclose(batch_multi_Uuat(*args), batch_multi_Uuat_sv(*args), atol=1e-5)
    assert np.isclose(batch_multi_Uuat(*args)[1, 2],
                      U.Uuat_qml(args[0][1], args[1][2], args[2][2], args[3][2]), atol=1e-5)


if __name__ == "__main__":
    test_scalar_input(); test_vector_input(); test_bloch_batch(); print("backend OK")
