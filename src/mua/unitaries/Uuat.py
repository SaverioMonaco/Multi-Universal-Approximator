"""Single-qubit universal approximator (data re-uploading, Pérez-Salinas et al. 2021).

    U(x) = prod_i  RY(2 phi_i) RZ(2 (w_i . x + alpha_i)),    f(x) = <0| U^dag Z U |0>

Two backends with the same signature ``(x, p_phi, pp_w, p_alpha) -> <Z>``:

* ``Uuat``      exact 2-component state vector in pure JAX (fast, jit/vmap/grad friendly);
                the default everywhere in the package.
* ``Uuat_qml``  the same circuit as a PennyLane QNode (reference, and for drawing).

``x`` is a scalar (regression, then ``pp_w`` has shape (depth,)) or a vector of length m
(generation, then ``pp_w`` has shape (depth, m)).
"""
import jax
import jax.numpy as jnp


def Uuat(x, p_phi, pp_w, p_alpha):
    """<Z> after the depth = len(p_phi) layers, starting from |0>."""

    def layer(psi, p):
        phi, p_w, alpha = p
        t = jnp.dot(p_w, x) + alpha  # RZ(2t) = diag(e^{-it}, e^{+it})
        psi = psi * jnp.stack([jnp.exp(-1j * t), jnp.exp(1j * t)])
        c, s = jnp.cos(phi), jnp.sin(phi)  # RY(2 phi)
        return jnp.stack([c * psi[0] - s * psi[1], s * psi[0] + c * psi[1]]), None

    psi0 = jnp.array([1.0 + 0.0j, 0.0 + 0.0j])
    psi, _ = jax.lax.scan(layer, psi0, (p_phi, pp_w, p_alpha))
    return jnp.abs(psi[0]) ** 2 - jnp.abs(psi[1]) ** 2


def ansatz(p_x, p_phi, pp_w, p_alpha, wire=0):
    """PennyLane gates of the ansatz (same parameter order as ``Uuat``)."""
    import pennylane as qml

    for phi, p_w, alpha in zip(p_phi, pp_w, p_alpha):
        qml.RZ(2 * (jnp.dot(p_w, p_x) + alpha), wires=wire)
        qml.RY(2 * phi, wires=wire)


def _make_qnode():
    import pennylane as qml

    @qml.qnode(qml.device("default.qubit", wires=1), interface="jax")
    def Uuat_qml(x, p_phi, pp_w, p_alpha):
        ansatz(x, p_phi, pp_w, p_alpha)
        return qml.expval(qml.PauliZ(0))

    return Uuat_qml


def __getattr__(name):  # lazy: PennyLane is only needed for the reference / drawings
    if name == "Uuat_qml":
        return _make_qnode()
    raise AttributeError(name)
