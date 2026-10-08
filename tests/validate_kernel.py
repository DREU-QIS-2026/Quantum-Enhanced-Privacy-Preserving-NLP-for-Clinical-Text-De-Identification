import numpy as np

from qiskit.quantum_info import Statevector

from circuits.feature_maps import FeatureMapBuilder


def encode(feature_vector):
    """Build and simulate the manual ZZFeatureMap."""
    builder = FeatureMapBuilder(
        reps=1,
        entanglement="linear"
    )

    circuit = builder.build_manual(len(feature_vector))
    bound_circuit = circuit.assign_parameters(feature_vector)

    return Statevector.from_instruction(bound_circuit)


def kernel(state_a, state_b):
    """Quantum kernel based on state fidelity."""
    overlap = np.vdot(state_a.data, state_b.data)
    return abs(overlap) ** 2


def main():
    x = np.array([
        0.25,
        0.50,
        0.75,
        1.00
    ])

    y = np.array([
        1.00,
        0.75,
        0.50,
        0.25
    ])

    state_x = encode(x)
    state_y = encode(y)

    k_xx = kernel(state_x, state_x)
    k_xy = kernel(state_x, state_y)
    k_yx = kernel(state_y, state_x)
    k_yy = kernel(state_y, state_y)

    print("=" * 50)
    print("Quantum Kernel Validation")
    print("=" * 50)

    print(f"Feature dimension : {len(x)}")
    print(f"Qubits            : {len(x)}")
    print(f"Repetitions       : 1")

    print("\nFeature vector X:")
    print(x)

    print("\nFeature vector Y:")
    print(y)

    print("\nKernel values")
    print("-" * 50)
    print(f"K(X,X) = {k_xx:.12f}")
    print(f"K(X,Y) = {k_xy:.12f}")
    print(f"K(Y,X) = {k_yx:.12f}")
    print(f"K(Y,Y) = {k_yy:.12f}")

    print("\nValidation")
    print("-" * 50)

    diagonal_pass = (
        np.isclose(k_xx, 1.0, atol=1e-10)
        and np.isclose(k_yy, 1.0, atol=1e-10)
    )

    symmetry_pass = np.isclose(
        k_xy,
        k_yx,
        atol=1e-10
    )

    range_pass = all(
        0.0 <= value <= 1.0 + 1e-10
        for value in [k_xx, k_xy, k_yx, k_yy]
    )

    print(f"Diagonal K(X,X), K(Y,Y) = {'PASS' if diagonal_pass else 'FAIL'}")
    print(f"Symmetry K(X,Y)=K(Y,X)  = {'PASS' if symmetry_pass else 'FAIL'}")
    print(f"Kernel range [0,1]       = {'PASS' if range_pass else 'FAIL'}")

    if diagonal_pass and symmetry_pass and range_pass:
        print("\nPASS: Quantum kernel calculation is valid.")
    else:
        print("\nFAIL: Quantum kernel validation failed.")


if __name__ == "__main__":
    main()