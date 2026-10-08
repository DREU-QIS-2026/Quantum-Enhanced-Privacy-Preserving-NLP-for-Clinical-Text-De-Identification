"""
Repeated qubit-scaling benchmark.

Measures the project's manual ZZFeatureMap quantum kernel on:
    - Qiskit CPU
    - qsim CUDA GPU

A warm-up run is performed before timing and is excluded from results.

Raw measurements are saved to:
    results/logs/qubit_scaling.csv
"""

from pathlib import Path
import sys
import csv
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.quantum_kernel import QuantumKernel


# ------------------------------------------------------------
# Benchmark configuration
# ------------------------------------------------------------

QUBIT_COUNTS = [4, 6, 8, 10, 12, 14, 16]

NUM_SAMPLES = 8
NUM_TIMED_RUNS = 5

RNG_SEED = 42

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "qubit_scaling.csv"
)


# ------------------------------------------------------------
# Workload generation
# ------------------------------------------------------------

def make_features(num_qubits):
    """
    Generate deterministic feature vectors in [0, pi].
    """

    rng = np.random.default_rng(
        RNG_SEED + num_qubits
    )

    return rng.uniform(
        0.0,
        np.pi,
        size=(NUM_SAMPLES, num_qubits),
    )


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

def validate_matrix(matrix):
    """
    Validate basic quantum-kernel properties.
    """

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    return (
        matrix.shape
        == (NUM_SAMPLES, NUM_SAMPLES)
        and np.all(np.isfinite(matrix))
        and np.allclose(
            matrix,
            matrix.T,
            atol=1e-5,
        )
        and np.allclose(
            np.diag(matrix),
            1.0,
            atol=1e-5,
        )
        and np.all(
            (matrix >= -1e-6)
            & (matrix <= 1.0 + 1e-6)
        )
    )


# ------------------------------------------------------------
# Benchmark one backend
# ------------------------------------------------------------

def benchmark_backend(
    backend,
    feature_sets,
):
    """
    Run the repeated benchmark for one backend.
    """

    results = []

    print("\n" + "=" * 70)
    print(f"{backend.upper()} BENCHMARK")
    print("=" * 70)

    kernel = QuantumKernel(
        reps=1,
        backend=backend,
    )

    for num_qubits in QUBIT_COUNTS:

        features = feature_sets[num_qubits]

        print(
            f"\n{num_qubits} qubits"
        )

        # ----------------------------------------------------
        # Warm-up
        # ----------------------------------------------------

        warmup_start = time.perf_counter()

        warmup_matrix = kernel.matrix(
            features
        )

        warmup_time = (
            time.perf_counter()
            - warmup_start
        )

        warmup_valid = validate_matrix(
            warmup_matrix
        )

        print(
            f"  Warm-up : "
            f"{warmup_time:.6f} s"
        )

        print(
            f"  Valid   : "
            f"{warmup_valid}"
        )

        # ----------------------------------------------------
        # Timed runs
        # ----------------------------------------------------

        for run_number in range(
            1,
            NUM_TIMED_RUNS + 1,
        ):

            start = time.perf_counter()

            matrix = kernel.matrix(
                features
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            valid = validate_matrix(
                matrix
            )

            print(
                f"  Run {run_number}: "
                f"{elapsed:.6f} s"
            )

            results.append(
                {
                    "backend": backend,
                    "qubits": num_qubits,
                    "samples": NUM_SAMPLES,
                    "run": run_number,
                    "runtime_seconds": elapsed,
                    "valid": valid,
                }
            )

    return results


# ------------------------------------------------------------
# Save raw measurements
# ------------------------------------------------------------

def save_results(results):
    """
    Save raw benchmark measurements to CSV.
    """

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "backend",
        "qubits",
        "samples",
        "run",
        "runtime_seconds",
        "valid",
    ]

    with RESULTS_FILE.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nRaw results saved to:\n"
        f"{RESULTS_FILE}"
    )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

def print_summary(results):
    """
    Print mean/min/max/std runtime by backend and qubit count.
    """

    print("\n" + "=" * 70)
    print("REPEATED BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        f"{'Qubits':>6} "
        f"{'Backend':>8} "
        f"{'Mean (s)':>12} "
        f"{'Std (s)':>12} "
        f"{'Min (s)':>12} "
        f"{'Max (s)':>12}"
    )

    print("-" * 70)

    summary = {}

    for backend in ["cpu", "gpu"]:

        for num_qubits in QUBIT_COUNTS:

            values = [
                row["runtime_seconds"]
                for row in results
                if (
                    row["backend"] == backend
                    and row["qubits"] == num_qubits
                )
            ]

            values = np.asarray(
                values,
                dtype=float,
            )

            mean = np.mean(values)
            std = np.std(values, ddof=1)
            minimum = np.min(values)
            maximum = np.max(values)

            summary[
                (backend, num_qubits)
            ] = mean

            print(
                f"{num_qubits:6d} "
                f"{backend:>8} "
                f"{mean:12.6f} "
                f"{std:12.6f} "
                f"{minimum:12.6f} "
                f"{maximum:12.6f}"
            )

    # --------------------------------------------------------
    # Speedup
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CPU vs GPU SPEEDUP")
    print("=" * 70)

    print(
        f"{'Qubits':>6} "
        f"{'CPU Mean (s)':>15} "
        f"{'GPU Mean (s)':>15} "
        f"{'CPU/GPU':>12}"
    )

    print("-" * 70)

    for num_qubits in QUBIT_COUNTS:

        cpu_mean = summary[
            ("cpu", num_qubits)
        ]

        gpu_mean = summary[
            ("gpu", num_qubits)
        ]

        speedup = (
            cpu_mean
            / gpu_mean
        )

        print(
            f"{num_qubits:6d} "
            f"{cpu_mean:15.6f} "
            f"{gpu_mean:15.6f} "
            f"{speedup:12.3f}x"
        )

    return summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("Repeated Quantum Kernel Qubit Scaling Benchmark")
    print("=" * 70)

    print("\nConfiguration")
    print("-" * 70)
    print(f"Qubit counts : {QUBIT_COUNTS}")
    print(f"Samples      : {NUM_SAMPLES}")
    print(f"Timed runs   : {NUM_TIMED_RUNS}")
    print(f"RNG seed     : {RNG_SEED}")
    print("Feature map  : Manual ZZFeatureMap")
    print("Repetitions  : 1")
    print("Warm-up      : 1 run per qubit count")

    # --------------------------------------------------------
    # Generate all workloads once.
    # --------------------------------------------------------

    feature_sets = {
        num_qubits: make_features(
            num_qubits
        )
        for num_qubits in QUBIT_COUNTS
    }

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    cpu_results = benchmark_backend(
        "cpu",
        feature_sets,
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    gpu_results = benchmark_backend(
        "gpu",
        feature_sets,
    )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    results = (
        cpu_results
        + gpu_results
    )

    # --------------------------------------------------------
    # Save raw data
    # --------------------------------------------------------

    save_results(results)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = print_summary(
        results
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    all_valid = all(
        row["valid"]
        for row in results
    )

    print("\n" + "=" * 70)

    if all_valid:
        print(
            "PASS: All repeated CPU and GPU "
            "quantum-kernel measurements were valid."
        )
    else:
        print(
            "FAIL: One or more benchmark "
            "measurements failed validation."
        )

    print("=" * 70)

    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()