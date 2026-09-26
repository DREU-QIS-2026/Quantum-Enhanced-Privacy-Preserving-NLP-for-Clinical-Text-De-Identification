import numpy as np

from qiskit import transpile

from circuits.feature_maps import FeatureMapBuilder
from backends.backend_factory import BackendFactory


def main():
    num_qubits = 4

    feature_vector = np.array([
        0.25,
        0.50,
        0.75,
        1.00,
    ])

    builder = FeatureMapBuilder(
        reps=1,
        entanglement="linear",
    )

    circuit = builder.build_manual(num_qubits)

    bound_circuit = circuit.assign_parameters(
        feature_vector
    )

    backend = BackendFactory.create(
        backend="cpu",
        method="statevector",
    )

    bound_circuit.save_statevector()

    compiled = transpile(
        bound_circuit,
        backend,
    )

    result = backend.run(
        compiled
    ).result()

    state = result.get_statevector()

    probabilities = np.abs(state) ** 2

    print("=" * 60)
    print("Aer Statevector Test")
    print("=" * 60)

    print(f"Backend : {backend.name}")
    print(f"Device  : CPU")
    print(f"Qubits  : {num_qubits}")
    print(f"State dimension : {len(state)}")
    print(
        f"Normalization   : "
        f"{probabilities.sum():.12f}"
    )

    if np.isclose(
        probabilities.sum(),
        1.0,
        atol=1e-10,
    ):
        print("\nPASS: Aer statevector simulation is valid.")
    else:
        print("\nFAIL: Statevector is not normalized.")


if __name__ == "__main__":
    main()