import numpy as np

from simulation.quantum_kernel import QuantumKernel


def main():
    # Four 4-dimensional feature vectors.
    feature_vectors = np.array([
        [0.25, 0.50, 0.75, 1.00],
        [1.00, 0.75, 0.50, 0.25],
        [0.10, 0.30, 0.60, 0.90],
        [0.90, 0.60, 0.30, 0.10],
    ])

    kernel = QuantumKernel(
        reps=1,
        entanglement="linear",
        backend="cpu",
    )

    kernel_matrix = kernel.matrix(feature_vectors)

    print("=" * 60)
    print("4x4 Quantum Kernel Matrix Test")
    print("=" * 60)

    print(f"Number of samples : {len(feature_vectors)}")
    print(f"Feature dimension  : {feature_vectors.shape[1]}")
    print(f"Number of qubits   : {feature_vectors.shape[1]}")
    print(f"Feature-map reps   : {kernel.reps}")

    print("\nFeature vectors:")
    for i, vector in enumerate(feature_vectors):
        print(f"X{i} = {vector}")

    print("\nKernel matrix:")
    print("-" * 60)
    print(kernel_matrix)

    print("\nValidation")
    print("-" * 60)

    diagonal_pass = np.allclose(
        np.diag(kernel_matrix),
        1.0,
        atol=1e-10,
    )

    symmetry_pass = np.allclose(
        kernel_matrix,
        kernel_matrix.T,
        atol=1e-10,
    )

    range_pass = np.all(
        (kernel_matrix >= -1e-10)
        & (kernel_matrix <= 1.0 + 1e-10)
    )

    eigenvalues = np.linalg.eigvalsh(kernel_matrix)

    psd_pass = np.all(
        eigenvalues >= -1e-10
    )

    print(
        f"Diagonal K(X,X) = 1       : "
        f"{'PASS' if diagonal_pass else 'FAIL'}"
    )

    print(
        f"Symmetry K(i,j)=K(j,i)    : "
        f"{'PASS' if symmetry_pass else 'FAIL'}"
    )

    print(
        f"Kernel range [0,1]        : "
        f"{'PASS' if range_pass else 'FAIL'}"
    )

    print(
        f"Positive semidefinite     : "
        f"{'PASS' if psd_pass else 'FAIL'}"
    )

    print("\nEigenvalues:")
    print(eigenvalues)

    if all([
        diagonal_pass,
        symmetry_pass,
        range_pass,
        psd_pass,
    ]):
        print(
            "\nPASS: 4x4 quantum kernel matrix is valid."
        )
    else:
        print(
            "\nFAIL: Kernel matrix validation failed."
        )


if __name__ == "__main__":
    main()