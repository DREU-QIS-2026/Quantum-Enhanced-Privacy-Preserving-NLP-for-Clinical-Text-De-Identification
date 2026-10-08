"""
Repeated NinaPro application benchmark.

Measures the complete 102-sample NinaPro quantum-kernel workload
on the project's CPU and GPU backends.

Warm-up runs are excluded from the reported measurements.

Raw results:
    results/logs/ninapro_benchmark.csv
"""

from pathlib import Path
import sys
import csv
import time
import argparse
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.ninapro import NinaProProcessor
from simulation.quantum_kernel import QuantumKernel


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASET = (
    PROJECT_ROOT
    / "data"
    / "ninapro"
    / "DB2"
    / "s1"
    / "S1_E1_A1.mat"
)

NUM_TIMED_RUNS = 5

DEFAULT_RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "ninapro_benchmark.csv"
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the NinaPro quantum-kernel benchmark."
    )

    parser.add_argument(
        "--backend",
        choices=["cpu", "gpu", "both"],
        default="both",
        help="Backend to benchmark.",
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="CSV file for raw benchmark results.",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# Kernel validation
# ------------------------------------------------------------

def validate_matrix(matrix, expected_samples):
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    return (
        matrix.shape
        == (
            expected_samples,
            expected_samples,
        )
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
# Benchmark backend
# ------------------------------------------------------------

def benchmark_backend(
    backend,
    features,
):
    """
    Warm up and then perform repeated measurements.
    """

    print("\n" + "=" * 70)
    print(f"{backend.upper()} NINAPRO BENCHMARK")
    print("=" * 70)

    kernel = QuantumKernel(
        reps=1,
        backend=backend,
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warmup_start = time.perf_counter()

    warmup_matrix = kernel.matrix(
        features
    )

    warmup_time = (
        time.perf_counter()
        - warmup_start
    )

    warmup_valid = validate_matrix(
        warmup_matrix,
        len(features),
    )

    print(
        f"\nWarm-up time : "
        f"{warmup_time:.6f} seconds"
    )

    print(
        f"Warm-up valid: "
        f"{warmup_valid}"
    )

    # --------------------------------------------------------
    # Timed runs
    # --------------------------------------------------------

    results = []

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
            matrix,
            len(features),
        )

        print(
            f"Run {run_number}: "
            f"{elapsed:.6f} seconds "
            f"(valid={valid})"
        )

        results.append(
            {
                "backend": backend,
                "run": run_number,
                "samples": len(features),
                "qubits": features.shape[1],
                "runtime_seconds": elapsed,
                "valid": valid,
            }
        )

    return results


# ------------------------------------------------------------
# Save raw results
# ------------------------------------------------------------

def save_results(results, results_file):

    results_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "backend",
        "run",
        "samples",
        "qubits",
        "runtime_seconds",
        "valid",
    ]

    with results_file.open(
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
        f"{results_file}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    args = parse_args()

    print("=" * 70)
    print("NinaPro Application Benchmark")
    print("=" * 70)

    print("\nDataset")
    print("-" * 70)
    print(DATASET)

    # --------------------------------------------------------
    # Process NinaPro once.
    # --------------------------------------------------------

    processor = NinaProProcessor(
        DATASET
    )

    features, labels, metadata = (
        processor.process()
    )

    print("\nWorkload")
    print("-" * 70)
    print(
        f"Samples       : {features.shape[0]}"
    )
    print(
        f"Quantum features: {features.shape[1]}"
    )
    print(
        f"Kernel matrix : "
        f"{features.shape[0]} × "
        f"{features.shape[0]}"
    )
    print(
        f"Timed runs    : {NUM_TIMED_RUNS}"
    )
    print(
        "Warm-up       : 1 run per backend"
    )

    results = []

    cpu_results = []
    gpu_results = []

    if args.backend in ("cpu", "both"):
        cpu_results = benchmark_backend(
            "cpu",
            features,
        )
        results.extend(cpu_results)

    if args.backend in ("gpu", "both"):
        gpu_results = benchmark_backend(
            "gpu",
            features,
        )
        results.extend(gpu_results)

    save_results(results, args.results_file)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    if cpu_results:
        cpu_times = np.array(
            [
                row["runtime_seconds"]
                for row in cpu_results
            ]
        )

        cpu_mean = np.mean(cpu_times)
        cpu_std = np.std(
            cpu_times,
            ddof=1,
        )
        cpu_median = np.median(
            cpu_times
        )

        print("\n" + "=" * 70)
        print("NINAPRO CPU SUMMARY")
        print("=" * 70)

        print(
            f"CPU mean       : "
            f"{cpu_mean:.6f} seconds"
        )

        print(
            f"CPU std        : "
            f"{cpu_std:.6f} seconds"
        )

        print(
            f"CPU median     : "
            f"{cpu_median:.6f} seconds"
        )

    if gpu_results:
        gpu_times = np.array(
            [
                row["runtime_seconds"]
                for row in gpu_results
            ]
        )

        gpu_mean = np.mean(gpu_times)
        gpu_std = np.std(
            gpu_times,
            ddof=1,
        )
        gpu_median = np.median(
            gpu_times
        )

        print("\n" + "=" * 70)
        print("NINAPRO GPU SUMMARY")
        print("=" * 70)

        print(
            f"GPU mean       : "
            f"{gpu_mean:.6f} seconds"
        )

        print(
            f"GPU std        : "
            f"{gpu_std:.6f} seconds"
        )

        print(
            f"GPU median     : "
            f"{gpu_median:.6f} seconds"
        )

    if cpu_results and gpu_results:
        speedup = (
            cpu_mean
            / gpu_mean
        )

        print(
            f"CPU/GPU ratio  : "
            f"{speedup:.3f}x"
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
            "PASS: All NinaPro benchmark runs "
            "produced valid kernel matrices."
        )
    else:
        print(
            "FAIL: One or more NinaPro benchmark "
            "runs produced an invalid kernel matrix."
        )

    print("=" * 70)

    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()