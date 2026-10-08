import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cirq
import qsimcirq

from qiskit.quantum_info import Statevector
from circuits.feature_maps import FeatureMapBuilder


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

X = np.array([
    0.25,
    0.50,
    0.75,
    1.00,
])

Y = np.array([
    0.40,
    0.60,
    0.80,
    1.00,
])

REPS = 1


# ------------------------------------------------------------
# Qiskit reference
# ------------------------------------------------------------

def build_qiskit_state(feature_vector):
    """Build and simulate the project's manual ZZFeatureMap."""

    builder = FeatureMapBuilder(
        reps=REPS,
        entanglement="linear",
    )

    circuit = builder.build_manual(
        len(feature_vector)
    )

    bound = circuit.assign_parameters(
        feature_vector
    )

    return Statevector.from_instruction(bound)


# ------------------------------------------------------------
# Cirq / qsim circuit
# ------------------------------------------------------------

def build_cirq_circuit(feature_vector):
    """Build the qsim equivalent of the manual ZZFeatureMap."""

    num_qubits = len(feature_vector)

    qubits = cirq.LineQubit.range(num_qubits)

    circuit = cirq.Circuit()

    for _ in range(REPS):

        # Hadamard layer
        for qubit in qubits:
            circuit.append(
                cirq.H(qubit)
            )

        # Feature encoding
        for i, qubit in enumerate(qubits):

            theta = 2 * feature_vector[i]

            circuit.append(
                cirq.ZPowGate(
                    exponent=theta / np.pi
                ).on(qubit)
            )

        # Linear ZZ entanglement
        for i in range(num_qubits - 1):

            q1 = qubits[i]
            q2 = qubits[i + 1]

            circuit.append(
                cirq.CNOT(q1, q2)
            )

            theta = (
                2
                * (np.pi - feature_vector[i])
                * (np.pi - feature_vector[i + 1])
            )

            circuit.append(
                cirq.ZPowGate(
                    exponent=theta / np.pi
                ).on(q2)
            )

            circuit.append(
                cirq.CNOT(q1, q2)
            )

    return circuit


def build_qsim_state(feature_vector):
    """Run the Cirq circuit using qsim CUDA."""

    circuit = build_cirq_circuit(
        feature_vector
    )

    options = qsimcirq.QSimOptions(
        use_gpu=True,
        gpu_mode=0,
    )

    simulator = qsimcirq.QSimSimulator(
        options
    )

    result = simulator.simulate(
        circuit
    )

    return np.asarray(
        result.final_state_vector,
        dtype=np.complex64,
    )


# ------------------------------------------------------------
# Qubit-order conversion
# ------------------------------------------------------------

def reorder_qsim_state(qsim_state, num_qubits):
    """
    Convert qsim/Cirq statevector ordering to
    Qiskit's computational-basis ordering.
    """

    reordered = np.zeros_like(qsim_state)

    for index in range(len(qsim_state)):

        reversed_index = int(
            format(
                index,
                f"0{num_qubits}b",
            )[::-1],
            2,
        )

        reordered[index] = qsim_state[
            reversed_index
        ]

    return reordered


# ------------------------------------------------------------
# Quantum kernel
# ------------------------------------------------------------

def kernel(state_a, state_b):
    """
    Fidelity quantum kernel:

        K(x, y) = |<psi(x) | psi(y)>|^2
    """

    overlap = np.vdot(
        state_a,
        state_b,
    )

    return float(
        abs(overlap) ** 2
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("Qiskit vs qsim GPU Quantum Kernel Validation")
    print("=" * 60)

    print(f"Features : {len(X)}")
    print(f"Qubits   : {len(X)}")
    print(f"Reps     : {REPS}")

    print("\nVector X:")
    print(X)

    print("\nVector Y:")
    print(Y)

    # --------------------------------------------------------
    # Qiskit
    # --------------------------------------------------------

    print("\nRunning Qiskit reference...")

    qiskit_x = build_qiskit_state(X)
    qiskit_y = build_qiskit_state(Y)

    qiskit_kernel = kernel(
        qiskit_x.data,
        qiskit_y.data,
    )

    # --------------------------------------------------------
    # qsim GPU
    # --------------------------------------------------------

    print("Running qsim CUDA GPU...")

    qsim_x = build_qsim_state(X)
    qsim_y = build_qsim_state(Y)

    qsim_x = reorder_qsim_state(
        qsim_x,
        len(X),
    )

    qsim_y = reorder_qsim_state(
        qsim_y,
        len(Y),
    )

    qsim_kernel = kernel(
        qsim_x,
        qsim_y,
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    difference = abs(
        qiskit_kernel - qsim_kernel
    )

    print("\nKernel comparison")
    print("-" * 60)

    print(
        f"Qiskit kernel : "
        f"{qiskit_kernel:.12f}"
    )

    print(
        f"qsim GPU kernel : "
        f"{qsim_kernel:.12f}"
    )

    print(
        f"Absolute difference : "
        f"{difference:.3e}"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if np.isclose(
        qiskit_kernel,
        qsim_kernel,
        atol=1e-6,
    ):
        print(
            "\nPASS: Qiskit and qsim GPU "
            "produce equivalent kernel values."
        )

    else:
        print(
            "\nFAIL: Qiskit and qsim GPU "
            "kernel values do not match."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()