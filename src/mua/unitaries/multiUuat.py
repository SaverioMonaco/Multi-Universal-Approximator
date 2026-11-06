"""Multi universal approximator: one independent ``Uuat`` per output (qubit), all fed the
same input noise, no entangling gates (Barthe et al. 2024)."""
from jax import vmap

from .Uuat import Uuat

# (x, (Q, D), (Q, D, m), (Q, D)) -> (Q,)
multi_Uuat = vmap(Uuat, in_axes=(None, 0, 0, 0))
# batch of noise vectors: ((B, m), ...) -> (B, Q)
batch_multi_Uuat = vmap(multi_Uuat, in_axes=(0, None, None, None))
