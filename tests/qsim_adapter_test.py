import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from qiskit.quantum_info import Statevector

from circuits.feature_maps import FeatureMapBuilder
from simulation.qsim_simulator import QSimSimulator


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
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("Project QSim Adapter Validation")
    print("=" * 60)

    print(f"Features : {len(FEATURE_VECTOR)}")
    print(f"Qubits   : {len(FEATURE_VECTOR)}")
    print(f"Reps     : {REPS}")

    # --------------------------------------------------------
    # Build project's manual circuit
    # --------------------------------------------------------

    builder = FeatureMapBuilder(
        reps=REPS,
        entanglement="linear",
    )

    circuit = builder.build_manual(
        len(FEATURE_VECTOR)
    )

    # --------------------------------------------------------
    # Qiskit reference
    # --------------------------------------------------------

    print("\nRunning Qiskit reference...")

    bound = circuit.assign_parameters(
        FEATURE_VECTOR
    )

    qiskit_state = Statevector.from_instruction(
        bound
    )

    # --------------------------------------------------------
    # Project qsim adapter
    # --------------------------------------------------------

    print("Running project qsim adapter...")

    simulator = QSimSimulator()

    qsim_state = simulator.statevector(
        circuit,
        FEATURE_VECTOR,
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    fidelity = abs(
        np.vdot(
            qiskit_state.data,
            qsim_state,
        )
    ) ** 2

    difference = abs(
        1.0 - fidelity
    )

    print("\nComparison")
    print("-" * 60)

    print(
        f"State fidelity : "
        f"{fidelity:.12f}"
    )

    print(
        f"Difference from 1 : "
        f"{difference:.3e}"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if np.isclose(
        fidelity,
        1.0,
        atol=1e-6,
    ):

        print(
            "\nPASS: Project qsim adapter produces "
            "the same quantum state as Qiskit."
        )

    else:

        print(
            "\nFAIL: Project qsim adapter does not "
            "match the Qiskit reference."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()