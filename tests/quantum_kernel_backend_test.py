import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from simulation.quantum_kernel import QuantumKernel


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


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("QuantumKernel CPU vs qsim GPU Validation")
    print("=" * 60)

    print("\nVector X:")
    print(X)

    print("\nVector Y:")
    print(Y)

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    print("\nRunning QuantumKernel CPU...")

    cpu_kernel = QuantumKernel(
        reps=1,
        entanglement="linear",
        backend="cpu",
    )

    cpu_value = cpu_kernel.evaluate_vectors(
        X,
        Y,
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    print("Running QuantumKernel GPU...")

    gpu_kernel = QuantumKernel(
        reps=1,
        entanglement="linear",
        backend="gpu",
    )

    gpu_value = gpu_kernel.evaluate_vectors(
        X,
        Y,
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    difference = abs(
        cpu_value - gpu_value
    )

    print("\nKernel comparison")
    print("-" * 60)

    print(
        f"CPU kernel : "
        f"{cpu_value:.12f}"
    )

    print(
        f"GPU kernel : "
        f"{gpu_value:.12f}"
    )

    print(
        f"Absolute difference : "
        f"{difference:.3e}"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if np.isclose(
        cpu_value,
        gpu_value,
        atol=1e-6,
    ):

        print(
            "\nPASS: QuantumKernel CPU and GPU "
            "produce equivalent kernel values."
        )

    else:

        print(
            "\nFAIL: QuantumKernel CPU and GPU "
            "kernel values do not match."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()