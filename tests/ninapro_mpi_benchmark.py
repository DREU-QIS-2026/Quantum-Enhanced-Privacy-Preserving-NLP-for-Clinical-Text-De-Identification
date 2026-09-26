"""
NinaPro real-data MPI quantum-kernel benchmark.

Strong scaling:
    Uses the actual 102-sample NinaPro workload.

        102 samples
            ↓
        8 quantum features
            ↓
        manual ZZFeatureMap
            ↓
        quantum-kernel matrix

    MPI ranks divide:
        1. Statevector encoding
        2. Upper-triangle kernel-pair evaluations

Weak scaling:
    Uses a NinaPro-derived workload extension so that the
    number of kernel-pair evaluations per MPI rank remains
    approximately constant as MPI process count increases.

    1 rank  -> 102 samples
    2 ranks -> 144 samples
    4 ranks -> 204 samples
    8 ranks -> 289 samples

    The 144-, 204-, and 289-sample workloads are deterministic
    extensions of the real 102-sample NinaPro feature set.
    They are computational scaling workloads, not additional
    independent biological observations.

This is MPI workload parallelism for quantum-kernel
evaluation. It is NOT distributed statevector simulation.

Results:
    results/logs/ninapro_mpi_strong_scaling.csv
    results/logs/ninapro_mpi_weak_scaling.csv
"""

from pathlib import Path
import argparse
import csv
import sys
import time

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

STRONG_RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "ninapro_mpi_strong_scaling.csv"
)

WEAK_RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "ninapro_mpi_weak_scaling.csv"
)

NUM_ITERATIONS = 5

SUPPORTED_WEAK_PROCESSES = {
    1: 102,
    2: 144,
    4: 204,
    8: 289,
}


# ------------------------------------------------------------
# Work splitting
# ------------------------------------------------------------

def split_work(
    total_items,
    rank,
    size,
):
    """
    Divide work as evenly as possible.
    """

    base = total_items // size
    remainder = total_items % size

    start = (
        rank * base
        + min(rank, remainder)
    )

    end = start + base

    if rank < remainder:
        end += 1

    return start, end


def upper_triangle_pairs(n):
    """
    Return all i <= j kernel pairs.
    """

    return [
        (i, j)
        for i in range(n)
        for j in range(i, n)
    ]


# ------------------------------------------------------------
# Weak-scaling workload
# ------------------------------------------------------------

def build_weak_scaling_features(
    features,
    target_samples,
):
    """
    Create a deterministic NinaPro-derived workload with
    target_samples feature vectors.

    The original 102 real feature vectors are reused in a
    deterministic cyclic order.

    This is intended only for computational weak-scaling
    measurements. It must not be interpreted as additional
    independent NinaPro observations.
    """

    source_samples = features.shape[0]

    indices = (
        np.arange(target_samples)
        % source_samples
    )

    return np.asarray(
        features[indices],
        dtype=float,
    )


# ------------------------------------------------------------
# Kernel validation
# ------------------------------------------------------------

def validate_matrix(matrix):

    if matrix is None:
        return False

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if matrix.ndim != 2:
        return False

    n = matrix.shape[0]

    return (
        matrix.shape == (n, n)
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
            (matrix >= -1e-6)
            &
            (matrix <= 1.0 + 1e-6)
        )
    )


# ------------------------------------------------------------
# One MPI iteration
# ------------------------------------------------------------

def run_iteration(
    features,
    kernel,
    comm,
    pairs,
):
    """
    Run one complete distributed kernel evaluation.

    Timed region includes:
        - statevector encoding
        - MPI state exchange
        - kernel pair evaluation
        - result gathering

    The pair list is supplied by the caller so it does not
    need to be regenerated for every timed iteration.
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    num_samples = len(features)

    # --------------------------------------------------------
    # Encode samples
    # --------------------------------------------------------

    sample_start, sample_end = split_work(
        num_samples,
        rank,
        size,
    )

    local_features = features[
        sample_start:sample_end
    ]

    local_states = []

    for feature_vector in local_features:

        state = kernel.encode(
            feature_vector
        )

        local_states.append(
            np.asarray(
                state.data,
                dtype=complex,
            )
        )

    state_dimension = (
        2 ** features.shape[1]
    )

    if local_states:

        local_states = np.asarray(
            local_states,
            dtype=complex,
        )

    else:

        local_states = np.empty(
            (
                0,
                state_dimension,
            ),
            dtype=complex,
        )

    # --------------------------------------------------------
    # Exchange states
    # --------------------------------------------------------

    gathered_states = comm.allgather(
        local_states
    )

    states = np.vstack(
        gathered_states
    )

    # --------------------------------------------------------
    # Divide upper-triangle kernel pairs
    # --------------------------------------------------------

    pair_start, pair_end = split_work(
        len(pairs),
        rank,
        size,
    )

    local_pairs = pairs[
        pair_start:pair_end
    ]

    # --------------------------------------------------------
    # Evaluate local kernel pairs
    # --------------------------------------------------------

    local_values = []

    for i, j in local_pairs:

        overlap = np.vdot(
            states[i],
            states[j],
        )

        value = float(
            abs(overlap) ** 2
        )

        local_values.append(
            value
        )

    # --------------------------------------------------------
    # Gather pair results
    # --------------------------------------------------------

    gathered_values = comm.gather(
        local_values,
        root=0,
    )

    # --------------------------------------------------------
    # Reconstruct matrix
    # --------------------------------------------------------

    matrix = None

    if rank == 0:

        matrix = np.zeros(
            (
                num_samples,
                num_samples,
            ),
            dtype=float,
        )

        offset = 0

        for rank_values in gathered_values:

            for value in rank_values:

                i, j = pairs[offset]

                matrix[i, j] = value
                matrix[j, i] = value

                offset += 1

    return matrix


# ------------------------------------------------------------
# CSV output
# ------------------------------------------------------------

def save_result(
    results_file,
    mode,
    processes,
    source_samples,
    workload_samples,
    features,
    pairs,
    iterations,
    iteration_times,
):
    """
    Append one benchmark result to a CSV file.
    """

    results_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = results_file.exists()

    mean_time = float(
        np.mean(iteration_times)
    )

    std_time = float(
        np.std(
            iteration_times,
            ddof=1,
        )
    )

    pair_count = len(pairs)

    pairs_per_process = (
        pair_count / processes
    )

    with results_file.open(
        "a",
        newline="",
    ) as file:

        writer = csv.writer(
            file
        )

        if not file_exists:

            writer.writerow([
                "timestamp",
                "scaling_mode",
                "processes",
                "source_samples",
                "workload_samples",
                "features",
                "qubits",
                "kernel_pairs",
                "pairs_per_process",
                "condition",
                "backend",
                "iterations",
                "mean_runtime_seconds",
                "std_runtime_seconds",
                "min_runtime_seconds",
                "max_runtime_seconds",
            ])

        writer.writerow([
            time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            mode,
            processes,
            source_samples,
            workload_samples,
            features.shape[1],
            features.shape[1],
            pair_count,
            pairs_per_process,
            "noiseless",
            "cpu",
            iterations,
            mean_time,
            std_time,
            float(
                np.min(
                    iteration_times
                )
            ),
            float(
                np.max(
                    iteration_times
                )
            ),
        ])

    return mean_time, std_time


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    # Keep mpi4py out of module-level imports so that
    # --help and syntax checks can run without MPI installed.
    from mpi4py import MPI

    parser = argparse.ArgumentParser(
        description=(
            "NinaPro MPI quantum-kernel benchmark."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "strong",
            "weak",
        ],
        default="strong",
        help=(
            "Scaling experiment to run. "
            "'strong' uses the actual 102-sample workload; "
            "'weak' uses a NinaPro-derived workload extension."
        ),
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=NUM_ITERATIONS,
        help=(
            "Number of timed kernel evaluations."
        ),
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help=(
            "CSV output path. If omitted, the default "
            "strong- or weak-scaling path is used."
        ),
    )

    args = parser.parse_args()

    if args.iterations < 1:
        parser.error(
            "--iterations must be at least 1"
        )

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    # --------------------------------------------------------
    # Validate weak-scaling process count
    # --------------------------------------------------------

    if (
        args.mode == "weak"
        and size not in SUPPORTED_WEAK_PROCESSES
    ):

        if rank == 0:

            print(
                "ERROR: Weak scaling supports only "
                "1, 2, or 4 MPI processes."
            )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Select results file
    # --------------------------------------------------------

    if args.results_file is not None:

        results_file = (
            args.results_file
        )

    elif args.mode == "strong":

        results_file = (
            STRONG_RESULTS_FILE
        )

    else:

        results_file = (
            WEAK_RESULTS_FILE
        )

    # --------------------------------------------------------
    # Load real NinaPro data on rank 0
    # --------------------------------------------------------

    if rank == 0:

        processor = NinaProProcessor(
            DATASET
        )

        source_features, labels, metadata = (
            processor.process()
        )

        source_features = np.asarray(
            source_features,
            dtype=float,
        )

        source_samples = (
            source_features.shape[0]
        )

    else:

        source_features = None
        source_samples = None

    # --------------------------------------------------------
    # Broadcast source workload
    # --------------------------------------------------------

    source_features = comm.bcast(
        source_features,
        root=0,
    )

    source_samples = comm.bcast(
        source_samples,
        root=0,
    )

    # --------------------------------------------------------
    # Select scaling workload
    # --------------------------------------------------------

    if args.mode == "strong":

        features = source_features

    else:

        target_samples = (
            SUPPORTED_WEAK_PROCESSES[size]
        )

        features = (
            build_weak_scaling_features(
                source_features,
                target_samples,
            )
        )

    num_samples = features.shape[0]
    num_features = features.shape[1]

    # --------------------------------------------------------
    # Build pair list once
    # --------------------------------------------------------

    pairs = upper_triangle_pairs(
        num_samples
    )

    # --------------------------------------------------------
    # Print configuration
    # --------------------------------------------------------

    if rank == 0:

        print("=" * 70)

        if args.mode == "strong":

            print(
                "NinaPro MPI Strong-Scaling Benchmark"
            )

        else:

            print(
                "NinaPro MPI Weak-Scaling Benchmark"
            )

        print("=" * 70)

        print("\nConfiguration")
        print("-" * 70)

        print(
            f"Scaling mode  : {args.mode}"
        )

        print(
            f"MPI processes  : {size}"
        )

        print(
            f"Source samples : {source_samples}"
        )

        print(
            f"Workload samples: {num_samples}"
        )

        print(
            f"Features       : {num_features}"
        )

        print(
            f"Qubits         : {num_features}"
        )

        print(
            "Condition      : noiseless"
        )

        print(
            "Backend        : Qiskit CPU"
        )

        print(
            f"Kernel pairs   : {len(pairs)}"
        )

        print(
            f"Pairs/process  : "
            f"{len(pairs) / size:.2f}"
        )

        print(
            f"Iterations     : {args.iterations}"
        )

        if args.mode == "weak":

            print(
                "\nNOTE: Weak-scaling workloads above "
                "102 samples reuse the real NinaPro "
                "feature vectors deterministically."
            )

    # --------------------------------------------------------
    # Create CPU quantum kernel
    # --------------------------------------------------------

    kernel = QuantumKernel(
        reps=1,
        backend="cpu",
    )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    comm.Barrier()

    warmup_matrix = run_iteration(
        features,
        kernel,
        comm,
        pairs,
    )

    comm.Barrier()

    if rank == 0:

        if not validate_matrix(
            warmup_matrix
        ):

            raise RuntimeError(
                "Warm-up kernel matrix "
                "failed validation."
            )

        print(
            "\nWarm-up: valid kernel matrix"
        )

    # --------------------------------------------------------
    # Timed iterations
    # --------------------------------------------------------

    iteration_times = []

    for iteration in range(
        1,
        args.iterations + 1,
    ):

        comm.Barrier()

        start = MPI.Wtime()

        matrix = run_iteration(
            features,
            kernel,
            comm,
            pairs,
        )

        comm.Barrier()

        elapsed = (
            MPI.Wtime()
            - start
        )

        iteration_times.append(
            elapsed
        )

        if rank == 0:

            valid = validate_matrix(
                matrix
            )

            print(
                f"Iteration {iteration}: "
                f"{elapsed:.6f} s "
                f"(valid={valid})"
            )

            if not valid:

                raise RuntimeError(
                    "Kernel matrix failed validation."
                )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    if rank == 0:

        mean_time, std_time = save_result(
            results_file=results_file,
            mode=args.mode,
            processes=size,
            source_samples=source_samples,
            workload_samples=num_samples,
            features=features,
            pairs=pairs,
            iterations=args.iterations,
            iteration_times=iteration_times,
        )

        print("\n" + "=" * 70)
        print("MPI SUMMARY")
        print("=" * 70)

        print(
            f"Scaling mode : {args.mode}"
        )

        print(
            f"Processes    : {size}"
        )

        print(
            f"Samples      : {num_samples}"
        )

        print(
            f"Kernel pairs : {len(pairs)}"
        )

        print(
            f"Pairs/rank   : "
            f"{len(pairs) / size:.2f}"
        )

        print(
            f"Mean time    : "
            f"{mean_time:.6f} s"
        )

        print(
            f"Std time     : "
            f"{std_time:.6f} s"
        )

        print(
            f"Min time     : "
            f"{np.min(iteration_times):.6f} s"
        )

        print(
            f"Max time     : "
            f"{np.max(iteration_times):.6f} s"
        )

        print(
            f"\nResults saved to:\n"
            f"{results_file}"
        )


if __name__ == "__main__":
    main()