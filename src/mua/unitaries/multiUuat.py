"""Multi universal approximator: one independent ``Uuat`` per output (qubit), all fed the
same input noise, no entangling gates (Barthe et al. 2024).

``batch_multi_Uuat(z, phi, w, alpha)``: ((B, m), (Q, D), (Q, D, m), (Q, D)) -> (B, Q) <Z>.

A single qubit stays pure, so instead of the complex state vector it is simulated as a real
Bloch vector r = (x, y, z), r0 = (0, 0, 1), <Z> = z:
    RZ(2t): rotation by 2t about the z axis,   RY(2 phi): rotation by 2 phi about the y axis.
All the re-uploading angles w . z + alpha are computed first in one tensor contraction, the
scan over the layers is then elementwise on (B, Q) arrays (about 3x faster than the vmapped
state vector on a GPU). ``batch_multi_Uuat_sv`` is the vmapped state-vector version
(reference, tests/test_backend.py)."""
import jax
import jax.numpy as jnp
from jax import vmap

from .Uuat import Uuat

# state-vector reference: (x, (Q, D), (Q, D, m), (Q, D)) -> (Q,), then over a batch of x
multi_Uuat = vmap(Uuat, in_axes=(None, 0, 0, 0))
batch_multi_Uuat_sv = vmap(multi_Uuat, in_axes=(0, None, None, None))


def batch_multi_Uuat(z, p_phi, pp_w, p_alpha):
    """<Z> of every qubit for a batch of noise vectors, (B, m) -> (B, Q)."""
    t = 2 * (jnp.einsum("qdm,bm->dbq", pp_w, z) + p_alpha.T[:, None, :])  # (D, B, Q)
    f = 2 * p_phi.T[:, None, :]                                             # (D, 1, Q)

    def layer(r, a):
        t, f = a
        x, y, zz = r
        ct, st, cf, sf = jnp.cos(t), jnp.sin(t), jnp.cos(f), jnp.sin(f)
        x, y = x * ct - y * st, x * st + y * ct          # RZ(2t)
        x, zz = x * cf + zz * sf, -x * sf + zz * cf      # RY(2 phi)
        return (x, y, zz), None

    zero = jnp.zeros(t.shape[1:], t.dtype)
    (_, _, zz), _ = jax.lax.scan(layer, (zero, zero, zero + 1), (t, f))
    return zz
