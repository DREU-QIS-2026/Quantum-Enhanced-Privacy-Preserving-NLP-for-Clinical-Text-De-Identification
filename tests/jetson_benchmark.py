import csv
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from circuits.feature_maps import FeatureMapBuilder
from backends.backend_factory import BackendFactory
from simulation.quantum_kernel import QuantumKernel


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

QUBITS = 4
REPS = 1

WARMUP_RUNS = 3
BENCHMARK_RUNS = 10

FEATURE_VECTOR = np.array([
    0.25,
    0.50,
    0.75,
    1.00,
])

# Small deterministic workload for kernel testing.
# Each vector has one feature per qubit.
KERNEL_FEATURE_VECTORS = np.array([
    [0.10, 0.20, 0.30, 0.40],
    [0.20, 0.30, 0.40, 0.50],
    [0.40, 0.50, 0.60, 0.70],
    [0.70, 0.80, 0.90, 1.00],
])

RESULTS_DIR = Path("results") / "jetson"


# ------------------------------------------------------------
# Environment information
# ------------------------------------------------------------

def get_command_output(command):
    """Run a command and return its output, or 'Unavailable'."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            return result.stdout.strip()

        return "Unavailable"

    except Exception:
        return "Unavailable"


def collect_environment():
    """Collect useful hardware/software information."""

    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "cpu": platform.processor(),
        "jetson_model": get_command_output(
            ["bash", "-c", "cat /proc/device-tree/model"]
        ),
        "ubuntu": get_command_output(
            ["bash", "-c", ". /etc/os-release && echo $PRETTY_NAME"]
        ),
        "l4t": get_command_output(
            ["bash", "-c", "cat /etc/nv_tegra_release"]
        ),
        "cuda": get_command_output(
            ["bash", "-c", "nvcc --version"]
        ),
    }


# ------------------------------------------------------------
# Circuit construction
# ------------------------------------------------------------

def build_circuit():
    """Build the project's manual ZZFeatureMap."""

    builder = FeatureMapBuilder(
        reps=REPS,
        entanglement="linear",
    )

    circuit = builder.build_manual(QUBITS)

    return circuit.assign_parameters(FEATURE_VECTOR)


# ------------------------------------------------------------
# Aer backend
# ------------------------------------------------------------

def create_backend(device):
    """
    Create the requested Aer backend.

    Supported values for this benchmark are currently:
        CPU
        GPU
    """

    if device.upper() == "CPU":
        return BackendFactory.create(
            backend="cpu",
            method="statevector",
        )

    if device.upper() == "GPU":
        return BackendFactory.create(
            backend="gpu",
            method="statevector",
        )

    raise ValueError(f"Unsupported device: {device}")


def run_once(backend, circuit):
    """
    Execute one statevector simulation.

    Returns elapsed wall-clock time in seconds.
    """

    start = time.perf_counter()

    compiled = backend.run(circuit)
    compiled.result()

    end = time.perf_counter()

    return end - start


# ------------------------------------------------------------
# Circuit benchmark
# ------------------------------------------------------------

def benchmark_device(device):
    """Benchmark one Aer device."""

    print()
    print("=" * 60)
    print(f"{device} CIRCUIT BENCHMARK")
    print("=" * 60)

    backend = create_backend(device)

    circuit = build_circuit()

    # Explicitly save the statevector so the simulation
    # actually produces the requested state.
    circuit.save_statevector()

    print(f"Backend          : {backend.name}")
    print(f"Device           : {device}")
    print(f"Qubits           : {QUBITS}")
    print(f"Feature-map reps : {REPS}")
    print(f"Warmup runs      : {WARMUP_RUNS}")
    print(f"Benchmark runs   : {BENCHMARK_RUNS}")

    # Warmup
    print("\nRunning warmup...")

    for _ in range(WARMUP_RUNS):
        run_once(backend, circuit)

    # Timed runs
    print("Running benchmark...")

    timings = []

    for i in range(BENCHMARK_RUNS):
        elapsed = run_once(
            backend,
            circuit,
        )

        timings.append(elapsed)

        print(
            f"Run {i + 1:2d}: "
            f"{elapsed:.6f} seconds"
        )

    mean_time = statistics.mean(timings)
    std_time = (
        statistics.stdev(timings)
        if len(timings) > 1
        else 0.0
    )

    min_time = min(timings)
    max_time = max(timings)

    print("\nResults")
    print("-" * 60)
    print(f"Mean : {mean_time:.6f} s")
    print(f"Std  : {std_time:.6f} s")
    print(f"Min  : {min_time:.6f} s")
    print(f"Max  : {max_time:.6f} s")

    return {
        "benchmark_type": "circuit",
        "device": device,
        "backend": backend.name,
        "qubits": QUBITS,
        "reps": REPS,
        "samples": 1,
        "kernel_pairs": "",
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "mean_seconds": mean_time,
        "std_seconds": std_time,
        "min_seconds": min_time,
        "max_seconds": max_time,
    }


# ------------------------------------------------------------
# Quantum kernel workload
# ------------------------------------------------------------

def run_kernel_once(kernel):
    """
    Build one complete quantum kernel matrix.

    Returns:
        elapsed time
        kernel matrix
    """

    start = time.perf_counter()

    kernel_matrix = kernel.matrix(
        KERNEL_FEATURE_VECTORS
    )

    end = time.perf_counter()

    return end - start, kernel_matrix


def validate_kernel_matrix(kernel_matrix):
    """
    Check basic kernel-matrix correctness.

    Expected properties:
        - square
        - symmetric
        - diagonal approximately 1
        - values approximately within [0, 1]
    """

    if kernel_matrix.shape != (
        len(KERNEL_FEATURE_VECTORS),
        len(KERNEL_FEATURE_VECTORS),
    ):
        return False

    if not np.allclose(
        kernel_matrix,
        kernel_matrix.T,
        atol=1e-10,
    ):
        return False

    if not np.allclose(
        np.diag(kernel_matrix),
        np.ones(len(KERNEL_FEATURE_VECTORS)),
        atol=1e-10,
    ):
        return False

    if np.any(kernel_matrix < -1e-10):
        return False

    if np.any(kernel_matrix > 1.0 + 1e-10):
        return False

    return True


def benchmark_kernel():
    """
    Benchmark the project's quantum-kernel workload.

    IMPORTANT:
        This uses the current QuantumKernel implementation.
        Its statevector path is the project's reference
        implementation, so this is a kernel-workload benchmark,
        not a GPU kernel benchmark.
    """

    print()
    print("=" * 60)
    print("QUANTUM KERNEL WORKLOAD")
    print("=" * 60)

    print(f"Qubits           : {QUBITS}")
    print(f"Feature-map reps : {REPS}")
    print(
        f"Samples          : "
        f"{len(KERNEL_FEATURE_VECTORS)}"
    )

    n_samples = len(KERNEL_FEATURE_VECTORS)

    kernel_pairs = (
        n_samples * (n_samples + 1) // 2
    )

    print(f"Kernel pairs     : {kernel_pairs}")
    print(f"Warmup runs      : {WARMUP_RUNS}")
    print(f"Benchmark runs   : {BENCHMARK_RUNS}")

    kernel = QuantumKernel(
        reps=REPS,
        entanglement="linear",
        backend="cpu",
    )

    # --------------------------------------------------------
    # First execution: correctness check
    # --------------------------------------------------------

    print("\nRunning kernel correctness check...")

    _, kernel_matrix = run_kernel_once(kernel)

    valid = validate_kernel_matrix(
        kernel_matrix
    )

    if valid:
        print("Kernel matrix: PASS")
    else:
        print("Kernel matrix: FAIL")

        print("\nKernel matrix:")
        print(kernel_matrix)

        raise RuntimeError(
            "Quantum kernel matrix failed validation."
        )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print("\nRunning kernel warmup...")

    for _ in range(WARMUP_RUNS):
        run_kernel_once(kernel)

    # --------------------------------------------------------
    # Timed runs
    # --------------------------------------------------------

    print("Running kernel benchmark...")

    timings = []

    for i in range(BENCHMARK_RUNS):

        elapsed, matrix = run_kernel_once(
            kernel
        )

        timings.append(elapsed)

        print(
            f"Run {i + 1:2d}: "
            f"{elapsed:.6f} seconds"
        )

    mean_time = statistics.mean(timings)

    std_time = (
        statistics.stdev(timings)
        if len(timings) > 1
        else 0.0
    )

    min_time = min(timings)
    max_time = max(timings)

    print("\nKernel Results")
    print("-" * 60)
    print(f"Mean : {mean_time:.6f} s")
    print(f"Std  : {std_time:.6f} s")
    print(f"Min  : {min_time:.6f} s")
    print(f"Max  : {max_time:.6f} s")

    print("\nKernel matrix:")
    print(kernel_matrix)

    return {
        "benchmark_type": "kernel_matrix_reference",
        "device": "reference",
        "backend": "reference_statevector",
        "qubits": QUBITS,
        "reps": REPS,
        "samples": n_samples,
        "kernel_pairs": kernel_pairs,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "mean_seconds": mean_time,
        "std_seconds": std_time,
        "min_seconds": min_time,
        "max_seconds": max_time,
    }


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

def save_results(environment, results):
    """Save benchmark results to CSV."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        RESULTS_DIR
        / f"jetson_benchmark_{timestamp}.csv"
    )

    rows = []

    for result in results:
        row = {
            **environment,
            **result,
        }

        rows.append(row)

    fieldnames = rows[0].keys()

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)
    print(output_file)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("JETSON / QUANTUM BENCHMARK")
    print("=" * 60)

    environment = collect_environment()

    print("\nEnvironment")
    print("-" * 60)

    for key, value in environment.items():
        print(f"{key:15}: {value}")

    results = []

    # --------------------------------------------------------
    # CPU circuit benchmark
    # --------------------------------------------------------

    try:
        results.append(
            benchmark_device("CPU")
        )

    except Exception as error:
        print()
        print("CPU CIRCUIT BENCHMARK FAILED")
        print("-" * 60)
        print(error)

    # --------------------------------------------------------
    # GPU circuit benchmark
    # --------------------------------------------------------

    try:
        results.append(
            benchmark_device("GPU")
        )

    except Exception as error:
        print()
        print("GPU CIRCUIT BENCHMARK FAILED")
        print("-" * 60)
        print(error)

    # --------------------------------------------------------
    # Reference quantum-kernel workload
    # --------------------------------------------------------

    try:
        results.append(
            benchmark_kernel()
        )

    except Exception as error:
        print()
        print("QUANTUM KERNEL BENCHMARK FAILED")
        print("-" * 60)
        print(error)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if results:
        save_results(
            environment,
            results,
        )
    else:
        print()
        print("No benchmark results were produced.")


if __name__ == "__main__":
    main()