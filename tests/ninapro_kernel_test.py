"""
End-to-end NinaPro -> quantum kernel validation.

Uses:
    NinaPro DB2 / S1 / E1 / A1
    17 gestures x 6 repetitions = 102 samples
    8 quantum features / 8 qubits

Compares the project QuantumKernel on:
    CPU
    qsim CUDA GPU
"""

from pathlib import Path
import sys
import time

import numpy as np


# Allow running this file directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.ninapro import NinaProProcessor
from simulation.quantum_kernel import QuantumKernel


DATASET = (
    PROJECT_ROOT
    / "data"
    / "ninapro"
    / "DB2"
    / "s1"
    / "S1_E1_A1.mat"
)


def validate_kernel_matrix(name, matrix):
    """Validate basic kernel-matrix properties."""
    matrix = np.asarray(matrix, dtype=float)

    print(f"\n{name} validation")
    print("-" * 55)

    print(f"Shape              : {matrix.shape}")
    print(f"Minimum            : {matrix.min():.9f}")
    print(f"Maximum            : {matrix.max():.9f}")
    print(f"Diagonal mean      : {np.mean(np.diag(matrix)):.9f}")
    print(f"Symmetric          : {np.allclose(matrix, matrix.T, atol=1e-5)}")
    print(f"Diagonal ~= 1      : {np.allclose(np.diag(matrix), 1.0, atol=1e-5)}")
    print(f"Values in [0, 1]   : {np.all((matrix >= -1e-6) & (matrix <= 1 + 1e-6))}")
    print(f"All finite         : {np.all(np.isfinite(matrix))}")

    return (
        matrix.shape == (102, 102)
        and np.allclose(matrix, matrix.T, atol=1e-5)
        and np.allclose(np.diag(matrix), 1.0, atol=1e-5)
        and np.all((matrix >= -1e-6) & (matrix <= 1 + 1e-6))
        and np.all(np.isfinite(matrix))
    )


def main():
    print("=" * 70)
    print("NinaPro End-to-End Quantum Kernel Validation")
    print("=" * 70)

    print(f"\nDataset:")
    print(DATASET)

    # ------------------------------------------------------------
    # 1. Process NinaPro
    # ------------------------------------------------------------
    print("\nProcessing NinaPro...")
    processor = NinaProProcessor(DATASET)

    features, labels, metadata = processor.process()

    print(f"Features           : {features.shape}")
    print(f"Labels             : {labels.shape}")
    print(f"Metadata           : {len(metadata)}")
    print(f"Qubits/features    : {features.shape[1]}")

    # ------------------------------------------------------------
    # 2. CPU quantum kernel
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CPU Quantum Kernel")
    print("=" * 70)

    cpu_kernel = QuantumKernel(
        reps=1,
        backend="cpu",
    )

    cpu_start = time.perf_counter()
    cpu_matrix = cpu_kernel.matrix(features)
    cpu_time = time.perf_counter() - cpu_start

    print(f"\nCPU kernel time    : {cpu_time:.6f} seconds")

    cpu_pass = validate_kernel_matrix(
        "CPU kernel matrix",
        cpu_matrix,
    )

    # ------------------------------------------------------------
    # 3. GPU quantum kernel
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GPU Quantum Kernel (qsim CUDA)")
    print("=" * 70)

    gpu_kernel = QuantumKernel(
        reps=1,
        backend="gpu",
    )

    gpu_start = time.perf_counter()
    gpu_matrix = gpu_kernel.matrix(features)
    gpu_time = time.perf_counter() - gpu_start

    print(f"\nGPU kernel time    : {gpu_time:.6f} seconds")

    gpu_pass = validate_kernel_matrix(
        "GPU kernel matrix",
        gpu_matrix,
    )

    # ------------------------------------------------------------
    # 4. CPU vs GPU comparison
    # ------------------------------------------------------------
    max_difference = np.max(
        np.abs(cpu_matrix - gpu_matrix)
    )

    mean_difference = np.mean(
        np.abs(cpu_matrix - gpu_matrix)
    )

    print("\n" + "=" * 70)
    print("CPU vs GPU Comparison")
    print("=" * 70)

    print(f"CPU time           : {cpu_time:.6f} seconds")
    print(f"GPU time           : {gpu_time:.6f} seconds")
    print(f"Maximum difference : {max_difference:.9e}")
    print(f"Mean difference    : {mean_difference:.9e}")

    if gpu_time > 0:
        print(f"CPU/GPU ratio      : {cpu_time / gpu_time:.3f}x")

    equivalent = np.allclose(
        cpu_matrix,
        gpu_matrix,
        atol=1e-5,
        rtol=1e-5,
    )

    print(f"Numerically equal  : {equivalent}")

    # ------------------------------------------------------------
    # 5. Final result
    # ------------------------------------------------------------
    print("\n" + "=" * 70)

    if cpu_pass and gpu_pass and equivalent:
        print("PASS: NinaPro data successfully completed the")
        print("      end-to-end CPU and qsim CUDA quantum-kernel pipeline.")
    else:
        print("FAIL: End-to-end quantum-kernel validation failed.")

    print("=" * 70)

    if not (cpu_pass and gpu_pass and equivalent):
        raise SystemExit(1)


if __name__ == "__main__":
    main()