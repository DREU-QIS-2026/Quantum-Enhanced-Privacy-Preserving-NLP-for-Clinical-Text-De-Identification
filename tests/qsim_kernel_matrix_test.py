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

FEATURE_VECTORS = np.array([
    [0.25, 0.50, 0.75, 1.00],
    [0.40, 0.60, 0.80, 1.00],
    [0.10, 0.30, 0.70, 0.90],
    [0.55, 0.65, 0.85, 0.95],
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
# Build kernel matrix
# ------------------------------------------------------------

def build_kernel_matrix(states):
    """Build a symmetric quantum kernel matrix."""

    n_samples = len(states)

    matrix = np.zeros(
        (n_samples, n_samples),
        dtype=float,
    )

    for i in range(n_samples):

        for j in range(i, n_samples):

            value = kernel(
                states[i],
                states[j],
            )

            matrix[i, j] = value
            matrix[j, i] = value

    return matrix


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("Qiskit vs qsim GPU Quantum Kernel Matrix Validation")
    print("=" * 60)

    num_samples = len(FEATURE_VECTORS)
    num_features = FEATURE_VECTORS.shape[1]

    print(f"Samples  : {num_samples}")
    print(f"Features : {num_features}")
    print(f"Qubits   : {num_features}")
    print(f"Reps     : {REPS}")

    print("\nFeature vectors:")
    print(FEATURE_VECTORS)

    # --------------------------------------------------------
    # Qiskit
    # --------------------------------------------------------

    print("\nRunning Qiskit reference...")

    qiskit_states = [
        build_qiskit_state(vector)
        for vector in FEATURE_VECTORS
    ]

    qiskit_matrix = build_kernel_matrix(
        [state.data for state in qiskit_states]
    )

    # --------------------------------------------------------
    # qsim GPU
    # --------------------------------------------------------

    print("Running qsim CUDA GPU...")

    qsim_states = [
        build_qsim_state(vector)
        for vector in FEATURE_VECTORS
    ]

    qsim_states = [
        reorder_qsim_state(
            state,
            num_features,
        )
        for state in qsim_states
    ]

    qsim_matrix = build_kernel_matrix(
        qsim_states
    )

    # --------------------------------------------------------
    # Print matrices
    # --------------------------------------------------------

    print("\nQiskit kernel matrix")
    print("-" * 60)
    print(
        np.array2string(
            qiskit_matrix,
            precision=8,
            suppress_small=True,
        )
    )

    print("\nqsim GPU kernel matrix")
    print("-" * 60)
    print(
        np.array2string(
            qsim_matrix,
            precision=8,
            suppress_small=True,
        )
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    max_difference = np.max(
        np.abs(
            qiskit_matrix - qsim_matrix
        )
    )

    print("\nMatrix comparison")
    print("-" * 60)

    print(
        f"Maximum absolute difference : "
        f"{max_difference:.3e}"
    )

    # --------------------------------------------------------
    # Matrix properties
    # --------------------------------------------------------

    qiskit_diagonal = np.diag(
        qiskit_matrix
    )

    qsim_diagonal = np.diag(
        qsim_matrix
    )

    qiskit_symmetric = np.allclose(
        qiskit_matrix,
        qiskit_matrix.T,
        atol=1e-6,
    )

    qsim_symmetric = np.allclose(
        qsim_matrix,
        qsim_matrix.T,
        atol=1e-6,
    )

    qiskit_range = (
        np.all(qiskit_matrix >= -1e-6)
        and np.all(qiskit_matrix <= 1.0 + 1e-6)
    )

    qsim_range = (
        np.all(qsim_matrix >= -1e-6)
        and np.all(qsim_matrix <= 1.0 + 1e-6)
    )

    qiskit_diagonal_valid = np.allclose(
        qiskit_diagonal,
        1.0,
        atol=1e-6,
    )

    qsim_diagonal_valid = np.allclose(
        qsim_diagonal,
        1.0,
        atol=1e-6,
    )

    print("\nKernel matrix properties")
    print("-" * 60)

    print(
        f"Qiskit diagonal = 1 : "
        f"{qiskit_diagonal_valid}"
    )

    print(
        f"qsim diagonal = 1   : "
        f"{qsim_diagonal_valid}"
    )

    print(
        f"Qiskit symmetric     : "
        f"{qiskit_symmetric}"
    )

    print(
        f"qsim symmetric       : "
        f"{qsim_symmetric}"
    )

    print(
        f"Qiskit values [0,1]  : "
        f"{qiskit_range}"
    )

    print(
        f"qsim values [0,1]    : "
        f"{qsim_range}"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    matrices_match = np.allclose(
        qiskit_matrix,
        qsim_matrix,
        atol=1e-6,
    )

    if (
        matrices_match
        and qiskit_diagonal_valid
        and qsim_diagonal_valid
        and qiskit_symmetric
        and qsim_symmetric
        and qiskit_range
        and qsim_range
    ):
        print(
            "\nPASS: Qiskit and qsim GPU "
            "produce equivalent quantum kernel matrices."
        )

    else:
        print(
            "\nFAIL: Kernel matrix validation failed."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()