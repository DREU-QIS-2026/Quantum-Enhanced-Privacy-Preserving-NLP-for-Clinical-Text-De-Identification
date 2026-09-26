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

FEATURE_VECTOR = np.array([
    0.25,
    0.50,
    0.75,
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
# Cirq / qsim version
# ------------------------------------------------------------

def build_cirq_circuit(feature_vector):
    """
    Build the same manual ZZFeatureMap using Cirq gates.

    This mirrors FeatureMapBuilder.build_manual():
        H
        P(2*x_i)
        CX
        P(2*(pi-x_i)*(pi-x_j))
        CX
    """

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
        #
        # Qiskit P(theta) = diag(1, exp(i*theta))
        # Cirq ZPowGate exponent e gives
        # diag(1, exp(i*pi*e)).
        #
        # Therefore:
        #     P(theta) == cirq.ZPowGate(exponent=theta/pi)
        #
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
    """Simulate the Cirq circuit using qsim CUDA."""

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
# Global-phase-invariant comparison
# ------------------------------------------------------------

def fidelity(state_a, state_b):
    """
    Calculate state fidelity:

        |<a|b>|^2

    This is invariant to a global phase difference.
    """

    overlap = np.vdot(
        state_a,
        state_b,
    )

    return abs(overlap) ** 2


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("Qiskit vs qsim GPU ZZFeatureMap Validation")
    print("=" * 60)

    print(f"Features : {len(FEATURE_VECTOR)}")
    print(f"Qubits   : {len(FEATURE_VECTOR)}")
    print(f"Reps     : {REPS}")

    print("\nFeature vector:")
    print(FEATURE_VECTOR)

    # --------------------------------------------------------
    # Qiskit reference
    # --------------------------------------------------------

    print("\nRunning Qiskit reference...")

    qiskit_state = build_qiskit_state(
        FEATURE_VECTOR
    )

    # --------------------------------------------------------
    # qsim GPU
    # --------------------------------------------------------

    print("Running qsim CUDA GPU...")

    qsim_state = build_qsim_state(
        FEATURE_VECTOR
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    print("\nState dimensions")
    print("-" * 60)

    print(
        f"Qiskit : {len(qiskit_state.data)}"
    )

    print(
        f"qsim   : {len(qsim_state)}"
    )

    # Match Qiskit's qubit-order convention

    num_qubits = len(FEATURE_VECTOR)

    qsim_state_reordered = np.zeros_like(qsim_state)

    for index in range(len(qsim_state)):
        reversed_index = int(
            format(index, f"0{num_qubits}b")[::-1],
            2,
        )

        qsim_state_reordered[index] = qsim_state[reversed_index]


    fidelity_value = fidelity(
        qiskit_state.data,
        qsim_state_reordered,
    )

    print("\nComparison")
    print("-" * 60)

    print(
        f"State fidelity : "
        f"{fidelity_value:.12f}"
    )

    print(
        f"Difference from 1 : "
        f"{abs(1.0 - fidelity_value):.3e}"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if np.isclose(
        fidelity_value,
        1.0,
        atol=1e-6,
    ):
        print(
            "\nPASS: Qiskit and qsim GPU "
            "produce equivalent quantum states."
        )

    else:
        print(
            "\nFAIL: Qiskit and qsim GPU "
            "states do not match."
        )

        print("\nQiskit state:")
        print(qiskit_state.data)

        print("\nqsim state (reordered):")
        print(qsim_state_reordered)

        raise SystemExit(1)


if __name__ == "__main__":
    main()