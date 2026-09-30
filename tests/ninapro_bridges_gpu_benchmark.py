"""
Bridges-2 GPU benchmark for the NinaPro quantum-kernel workload.

This script is intentionally separate from the local GPU benchmark.
The local GPU path uses qsim CUDA, while this script uses Qiskit Aer
directly on Bridges-2.

Backends:
    cpu
        Aer statevector on CPU

    gpu
        Aer statevector on GPU

    tensor
        Aer tensor_network on GPU using the installed cuQuantum stack

Modes:
    ninapro
        Full 102-sample NinaPro quantum-kernel benchmark

    scaling
        Qubit-scaling benchmark at 4, 6, 8, 10, 12, 14, and 16 qubits

    both
        Run both benchmark families
"""

from pathlib import Path
import sys
import csv
import time
import argparse

import numpy as np

from qiskit import transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from circuits.feature_maps import FeatureMapBuilder
from datasets.ninapro import NinaProProcessor


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATASET = (
    PROJECT_ROOT
    / "data"
    / "ninapro"
    / "DB2"
    / "s1"
    / "S1_E1_A1.mat"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "logs"
)

NINAPRO_RESULTS_FILE = (
    RESULTS_DIR
    / "ninapro_bridges_gpu_benchmark.csv"
)

SCALING_RESULTS_FILE = (
    RESULTS_DIR
    / "qubit_scaling_bridges_gpu.csv"
)

QUBIT_COUNTS = [
    4,
    6,
    8,
    10,
    12,
    14,
    16,
]

SCALING_SAMPLES = 8
RNG_SEED = 42

DEFAULT_RUNS = 5


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the QuantumHPC workflow on Bridges-2 "
            "CPU and V100 GPU backends."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "ninapro",
            "scaling",
            "both",
        ],
        default="both",
        help="Benchmark family to execute.",
    )

    parser.add_argument(
        "--backend",
        choices=[
            "cpu",
            "gpu",
            "tensor",
            "all",
        ],
        default="all",
        help=(
            "Aer backend configuration. "
            "'gpu' uses statevector/GPU and "
            "'tensor' uses tensor_network/GPU."
        ),
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Number of timed runs after warm-up.",
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    return args


# ---------------------------------------------------------------------
# Backend construction
# ---------------------------------------------------------------------

def create_backend(backend_name):

    if backend_name == "cpu":

        backend = AerSimulator(
            method="statevector",
            device="CPU",
        )

        label = "cpu_statevector"

    elif backend_name == "gpu":

        backend = AerSimulator(
            method="statevector",
            device="GPU",
        )

        label = "gpu_statevector"

    elif backend_name == "tensor":

        backend = AerSimulator(
            method="tensor_network",
            device="GPU",
        )

        label = "gpu_tensor_network"

    else:

        raise ValueError(
            f"Unknown backend: {backend_name}"
        )

    return backend, label


def selected_backends(backend_argument):

    if backend_argument == "all":
        return [
            "cpu",
            "gpu",
            "tensor",
        ]

    return [
        backend_argument
    ]


# ---------------------------------------------------------------------
# Statevector simulation
# ---------------------------------------------------------------------

def simulate_state(
    parameterized_circuit,
    feature_vector,
    backend,
):

    bound_circuit = (
        parameterized_circuit
        .assign_parameters(
            feature_vector
        )
    )

    bound_circuit.save_statevector()

    compiled = transpile(
        bound_circuit,
        backend,
    )

    result = backend.run(
        compiled
    ).result()

    state = np.asarray(
        result.get_statevector(),
        dtype=np.complex128,
    )

    return state


# ---------------------------------------------------------------------
# Quantum-kernel construction
# ---------------------------------------------------------------------

def build_kernel_matrix(
    features,
    backend,
):

    features = np.asarray(
        features,
        dtype=float,
    )

    num_samples = features.shape[0]
    num_qubits = features.shape[1]

    builder = FeatureMapBuilder(
        reps=1,
        entanglement="linear",
    )

    circuit = builder.build_manual(
        num_qubits
    )

    states = []

    for feature_vector in features:

        state = simulate_state(
            circuit,
            feature_vector,
            backend,
        )

        states.append(
            state
        )

    kernel = np.zeros(
        (
            num_samples,
            num_samples,
        ),
        dtype=float,
    )

    for i in range(num_samples):

        for j in range(
            i,
            num_samples,
        ):

            overlap = np.vdot(
                states[i],
                states[j],
            )

            value = float(
                abs(overlap) ** 2
            )

            kernel[i, j] = value
            kernel[j, i] = value

    return kernel


# ---------------------------------------------------------------------
# Kernel validation
# ---------------------------------------------------------------------

def validate_kernel(
    matrix,
    expected_samples,
):

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
        and np.all(
            np.isfinite(matrix)
        )
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
            (
                matrix >= -1e-6
            )
            &
            (
                matrix <= 1.0 + 1e-6
            )
        )
    )


# ---------------------------------------------------------------------
# Benchmark one workload
# ---------------------------------------------------------------------

def benchmark_workload(
    workload_name,
    features,
    backend_name,
    runs,
):

    backend, backend_label = (
        create_backend(
            backend_name
        )
    )

    print()
    print("=" * 72)
    print(
        f"{workload_name} | "
        f"{backend_label}"
    )
    print("=" * 72)

    print(
        f"Method   : "
        f"{backend.options.method}"
    )

    print(
        f"Device   : "
        f"{backend.options.device}"
    )

    print(
        f"Samples  : "
        f"{features.shape[0]}"
    )

    print(
        f"Qubits   : "
        f"{features.shape[1]}"
    )

    print(
        f"Runs     : "
        f"{runs}"
    )

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    warmup_start = (
        time.perf_counter()
    )

    warmup_matrix = (
        build_kernel_matrix(
            features,
            backend,
        )
    )

    warmup_time = (
        time.perf_counter()
        - warmup_start
    )

    warmup_valid = (
        validate_kernel(
            warmup_matrix,
            len(features),
        )
    )

    print()
    print(
        f"Warm-up : "
        f"{warmup_time:.6f} s "
        f"(valid={warmup_valid})"
    )

    if not warmup_valid:

        raise RuntimeError(
            "Warm-up kernel "
            "failed validation."
        )

    # ---------------------------------------------------------
    # Timed runs
    # ---------------------------------------------------------

    rows = []

    for run_number in range(
        1,
        runs + 1,
    ):

        start = (
            time.perf_counter()
        )

        matrix = (
            build_kernel_matrix(
                features,
                backend,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        valid = (
            validate_kernel(
                matrix,
                len(features),
            )
        )

        print(
            f"Run {run_number}: "
            f"{elapsed:.6f} s "
            f"(valid={valid})"
        )

        if not valid:

            raise RuntimeError(
                "Timed kernel "
                "failed validation."
            )

        rows.append(
            {
                "workload": workload_name,
                "backend": backend_label,
                "method": str(
                    backend.options.method
                ),
                "device": str(
                    backend.options.device
                ),
                "samples": features.shape[0],
                "qubits": features.shape[1],
                "run": run_number,
                "runtime_seconds": elapsed,
                "valid": valid,
            }
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    times = np.asarray(
        [
            row["runtime_seconds"]
            for row in rows
        ],
        dtype=float,
    )

    print()
    print(
        f"Mean   : "
        f"{np.mean(times):.6f} s"
    )

    if len(times) > 1:

        print(
            f"Std    : "
            f"{np.std(times, ddof=1):.6f} s"
        )

    print(
        f"Median : "
        f"{np.median(times):.6f} s"
    )

    print(
        f"Min    : "
        f"{np.min(times):.6f} s"
    )

    print(
        f"Max    : "
        f"{np.max(times):.6f} s"
    )

    return rows


# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------

def save_results(
    rows,
    output_file,
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "workload",
        "backend",
        "method",
        "device",
        "samples",
        "qubits",
        "run",
        "runtime_seconds",
        "valid",
    ]

    with output_file.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        "Results saved to:"
    )

    print(
        output_file
    )


# ---------------------------------------------------------------------
# NinaPro workload
# ---------------------------------------------------------------------

def run_ninapro(
    backend_names,
    runs,
):

    print()
    print("=" * 72)
    print(
        "Loading NinaPro workload"
    )
    print("=" * 72)

    processor = NinaProProcessor(
        DATASET
    )

    features, labels, metadata = (
        processor.process()
    )

    features = np.asarray(
        features,
        dtype=float,
    )

    print(
        f"Feature matrix: "
        f"{features.shape}"
    )

    all_rows = []

    for backend_name in (
        backend_names
    ):

        rows = benchmark_workload(
            workload_name=(
                "ninapro_102x102"
            ),
            features=features,
            backend_name=backend_name,
            runs=runs,
        )

        all_rows.extend(
            rows
        )

    save_results(
        all_rows,
        NINAPRO_RESULTS_FILE,
    )


# ---------------------------------------------------------------------
# Qubit-scaling workload
# ---------------------------------------------------------------------

def make_scaling_features(
    num_qubits,
):

    rng = np.random.default_rng(
        RNG_SEED
        + num_qubits
    )

    return rng.uniform(
        0.0,
        np.pi,
        size=(
            SCALING_SAMPLES,
            num_qubits,
        ),
    )


def run_scaling(
    backend_names,
    runs,
):

    all_rows = []

    for num_qubits in (
        QUBIT_COUNTS
    ):

        features = (
            make_scaling_features(
                num_qubits
            )
        )

        workload_name = (
            f"scaling_{num_qubits}q"
        )

        for backend_name in (
            backend_names
        ):

            rows = (
                benchmark_workload(
                    workload_name=(
                        workload_name
                    ),
                    features=features,
                    backend_name=(
                        backend_name
                    ),
                    runs=runs,
                )
            )

            all_rows.extend(
                rows
            )

    save_results(
        all_rows,
        SCALING_RESULTS_FILE,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    args = parse_args()

    backends = (
        selected_backends(
            args.backend
        )
    )

    print("=" * 72)
    print(
        "Bridges-2 QuantumHPC GPU Benchmark"
    )
    print("=" * 72)

    print(
        f"Mode     : {args.mode}"
    )

    print(
        f"Backends : {backends}"
    )

    print(
        f"Runs     : {args.runs}"
    )

    print(
        "Feature map: "
        "manual ZZFeatureMap, reps=1"
    )

    if args.mode in (
        "ninapro",
        "both",
    ):

        run_ninapro(
            backends,
            args.runs,
        )

    if args.mode in (
        "scaling",
        "both",
    ):

        run_scaling(
            backends,
            args.runs,
        )

    print()
    print("=" * 72)
    print(
        "BENCHMARK COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()