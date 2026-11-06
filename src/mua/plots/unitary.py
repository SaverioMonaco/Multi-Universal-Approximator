"""Circuit drawings (PennyLane)."""
import numpy as np
from matplotlib.pyplot import savefig

from ..unitaries.Uuat import ansatz


def show_single(depth, save=""):
    """Single-qubit approximator with `depth` layers."""
    import pennylane as qml

    @qml.qnode(qml.device("default.qubit", wires=1))
    def circuit(x, p_phi, pp_w, p_alpha):
        ansatz(x, p_phi, pp_w, p_alpha)
        return qml.expval(qml.PauliZ(0))

    qml.draw_mpl(circuit)(np.array(0.0), np.arange(depth), np.ones(depth), np.arange(depth))
    if save:
        savefig(save)


def show_multi(num_qubits, depth, save=""):
    """Multi universal approximator: `num_qubits` independent approximators."""
    import pennylane as qml

    @qml.qnode(qml.device("default.qubit", wires=num_qubits))
    def circuit(x, p_phi, pp_w, p_alpha):
        for q in range(num_qubits):
            ansatz(x, p_phi, pp_w, p_alpha, wire=q)
        return [qml.expval(qml.PauliZ(q)) for q in range(num_qubits)]

    qml.draw_mpl(circuit)(np.zeros(depth), np.arange(depth), np.ones((depth, depth)),
                          np.arange(depth))
    if save:
        savefig(save)
