"""
Full-capacity real-data NinaPro MPI quantum-kernel benchmark.

This benchmark uses every complete, non-overlapping 512-sample
window available from the longest contiguous segment for each
gesture/repetition pair in NinaPro DB2 S1_E1_A1.mat.

Unlike the balanced 714-window benchmark, the number of windows
per segment is allowed to vary according to the amount of real
recorded data available.

Expected workload:

    17 gestures
    x 6 repetitions
    = 102 gesture/repetition segments

    7-27 complete windows per segment
    1404 total distinct real EMG windows

Each window:
    12 EMG channels
        ->
    MAV + RMS + ZC
        ->
    36 classical features
        ->
    StandardScaler
        ->
    PCA to 8 dimensions
        ->
    MinMaxScaler to [0, pi]
        ->
    8 quantum features

For 1404 samples:

    1404 * 1405 / 2 = 986310

upper-triangular quantum-kernel pairs.

MPI distributes:
    1. quantum-state encoding
    2. state exchange
    3. kernel-pair evaluation
    4. result gathering

This is workload-level MPI parallelism. It does not distribute
one quantum statevector across multiple nodes.
"""

from pathlib import Path
import argparse
import csv
import os
import sys
import time

import numpy as np


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.ninapro import NinaProProcessor
from simulation.quantum_kernel import QuantumKernel


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

DEFAULT_RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "ninapro_mpi_fullcapacity.csv"
)

NUM_ITERATIONS = 5


# ---------------------------------------------------------------------
# Work splitting
# ---------------------------------------------------------------------

def split_work(
    total_items,
    rank,
    size,
):
    """
    Divide a contiguous workload as evenly as possible
    among MPI ranks.
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
    Return all quantum-kernel pairs i <= j.
    """

    return [
        (i, j)
        for i in range(n)
        for j in range(i, n)
    ]


# ---------------------------------------------------------------------
# Full-capacity real-data workload
# ---------------------------------------------------------------------

def build_fullcapacity_features():
    """
    Build the maximum non-overlapping-window workload available
    from S1_E1_A1.mat.

    For every gesture/repetition pair:

        1. Find the longest contiguous real segment.
        2. Determine how many complete 512-sample windows fit.
        3. Center the complete-window block within the segment.
        4. Extract every non-overlapping complete window.

    No feature vector is duplicated or synthetically generated.
    """

    processor = NinaProProcessor(
        DATASET
    )

    emg, restimulus, rerepetition = (
        processor.load()
    )

    window_size = (
        processor.window_size
    )

    raw_features = []
    labels = []
    metadata = []

    windows_per_segment = []

    for gesture in (
        processor.GESTURE_LABELS
    ):

        for repetition in (
            processor.REPETITIONS
        ):

            start, end = (
                processor._find_segment(
                    restimulus,
                    rerepetition,
                    gesture,
                    repetition,
                )
            )

            segment_length = (
                end - start
            )

            num_windows = (
                segment_length
                // window_size
            )

            if num_windows < 1:

                raise ValueError(
                    "No complete window available for "
                    f"gesture={gesture}, "
                    f"repetition={repetition}"
                )

            windows_per_segment.append(
                num_windows
            )

            # -------------------------------------------------
            # Center the largest block containing only complete
            # non-overlapping windows.
            # -------------------------------------------------

            used_samples = (
                num_windows
                * window_size
            )

            block_start = (
                start
                + (
                    segment_length
                    - used_samples
                )
                // 2
            )

            for window_index in range(
                num_windows
            ):

                window_start = (
                    block_start
                    + window_index
                    * window_size
                )

                window_end = (
                    window_start
                    + window_size
                )

                window = emg[
                    window_start:window_end
                ]

                if (
                    window.shape[0]
                    != window_size
                ):

                    raise ValueError(
                        "Incorrect window size for "
                        f"gesture={gesture}, "
                        f"repetition={repetition}, "
                        f"window={window_index}"
                    )

                features = (
                    processor._extract_features(
                        window
                    )
                )

                raw_features.append(
                    features
                )

                labels.append(
                    gesture
                )

                metadata.append(
                    {
                        "gesture": gesture,
                        "repetition": repetition,
                        "window_index": window_index,
                        "window_start": window_start,
                        "window_end": window_end,
                        "segment_start": start,
                        "segment_end": end,
                        "segment_length": segment_length,
                    }
                )

    raw_features = np.asarray(
        raw_features,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    windows_per_segment = np.asarray(
        windows_per_segment,
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # Same preprocessing pipeline used by the existing
    # NinaPro workloads.
    # ---------------------------------------------------------

    standardized = (
        processor.scaler.fit_transform(
            raw_features
        )
    )

    reduced = (
        processor.pca.fit_transform(
            standardized
        )
    )

    quantum_features = (
        processor.quantum_scaler.fit_transform(
            reduced
        )
    )

    return (
        np.asarray(
            quantum_features,
            dtype=float,
        ),
        labels,
        metadata,
        windows_per_segment,
    )


# ---------------------------------------------------------------------
# Kernel validation
# ---------------------------------------------------------------------

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
# One distributed kernel evaluation
# ---------------------------------------------------------------------

def run_iteration(
    features,
    kernel,
    comm,
    pairs,
):

    rank = comm.Get_rank()
    size = comm.Get_size()

    num_samples = (
        len(features)
    )

    # ---------------------------------------------------------
    # Distribute statevector encoding
    # ---------------------------------------------------------

    sample_start, sample_end = (
        split_work(
            num_samples,
            rank,
            size,
        )
    )

    local_features = features[
        sample_start:sample_end
    ]

    local_states = []

    for feature_vector in (
        local_features
    ):

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

    # ---------------------------------------------------------
    # Exchange encoded states between MPI ranks
    # ---------------------------------------------------------

    gathered_states = (
        comm.allgather(
            local_states
        )
    )

    states = np.vstack(
        gathered_states
    )

    # ---------------------------------------------------------
    # Distribute upper-triangular quantum-kernel pairs
    # ---------------------------------------------------------

    pair_start, pair_end = (
        split_work(
            len(pairs),
            rank,
            size,
        )
    )

    local_pairs = pairs[
        pair_start:pair_end
    ]

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

    # ---------------------------------------------------------
    # Gather pair values
    # ---------------------------------------------------------

    gathered_values = (
        comm.gather(
            local_values,
            root=0,
        )
    )

    matrix = None

    # ---------------------------------------------------------
    # Reconstruct complete symmetric kernel matrix
    # ---------------------------------------------------------

    if rank == 0:

        matrix = np.zeros(
            (
                num_samples,
                num_samples,
            ),
            dtype=float,
        )

        offset = 0

        for rank_values in (
            gathered_values
        ):

            for value in (
                rank_values
            ):

                i, j = pairs[
                    offset
                ]

                matrix[i, j] = value
                matrix[j, i] = value

                offset += 1

    return matrix


# ---------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------

def save_result(
    results_file,
    processes,
    node_count,
    ranks_per_node,
    features,
    pairs,
    iterations,
    iteration_times,
    windows_per_segment,
):

    results_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mean_time = float(
        np.mean(
            iteration_times
        )
    )

    if len(iteration_times) > 1:

        std_time = float(
            np.std(
                iteration_times,
                ddof=1,
            )
        )

    else:

        std_time = 0.0

    pair_count = len(
        pairs
    )

    pairs_per_process = (
        pair_count
        / processes
    )

    file_exists = (
        results_file.exists()
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
                "processes",
                "nodes",
                "ranks_per_node",
                "segments",
                "min_windows_per_segment",
                "max_windows_per_segment",
                "mean_windows_per_segment",
                "real_windows",
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
            processes,
            node_count,
            ranks_per_node,
            len(windows_per_segment),
            int(
                np.min(
                    windows_per_segment
                )
            ),
            int(
                np.max(
                    windows_per_segment
                )
            ),
            float(
                np.mean(
                    windows_per_segment
                )
            ),
            len(features),
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

    return (
        mean_time,
        std_time,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    from mpi4py import MPI

    parser = argparse.ArgumentParser(
        description=(
            "Full-capacity real-data NinaPro "
            "MPI quantum-kernel benchmark."
        )
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=NUM_ITERATIONS,
        help=(
            "Number of timed iterations "
            "after one warm-up."
        ),
    )

    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="CSV output file.",
    )

    args = parser.parse_args()

    if args.iterations < 1:

        parser.error(
            "--iterations must be at least 1"
        )

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    # ---------------------------------------------------------
    # Build full real-data workload on rank 0
    # ---------------------------------------------------------

    if rank == 0:

        (
            features,
            labels,
            metadata,
            windows_per_segment,
        ) = build_fullcapacity_features()

    else:

        features = None
        windows_per_segment = None

    # ---------------------------------------------------------
    # Broadcast processed features and workload statistics
    # ---------------------------------------------------------

    features = comm.bcast(
        features,
        root=0,
    )

    windows_per_segment = comm.bcast(
        windows_per_segment,
        root=0,
    )

    num_samples = (
        features.shape[0]
    )

    num_features = (
        features.shape[1]
    )

    pairs = (
        upper_triangle_pairs(
            num_samples
        )
    )

    # ---------------------------------------------------------
    # Slurm topology information
    # ---------------------------------------------------------

    node_count = int(
        os.environ.get(
            "SLURM_JOB_NUM_NODES",
            "1",
        )
    )

    ranks_per_node = (
        size / node_count
    )

    # ---------------------------------------------------------
    # Configuration output
    # ---------------------------------------------------------

    if rank == 0:

        print("=" * 72)
        print(
            "NinaPro Full-Capacity "
            "Real-Data MPI Benchmark"
        )
        print("=" * 72)

        print()
        print("Configuration")
        print("-" * 72)

        print(
            f"MPI processes       : {size}"
        )

        print(
            f"Nodes               : {node_count}"
        )

        print(
            f"Ranks/node          : {ranks_per_node:.2f}"
        )

        print(
            f"Segments            : "
            f"{len(windows_per_segment)}"
        )

        print(
            "Windows/segment     : "
            f"{int(np.min(windows_per_segment))}-"
            f"{int(np.max(windows_per_segment))} "
            f"(mean "
            f"{np.mean(windows_per_segment):.2f})"
        )

        print(
            f"Real EMG windows    : {num_samples}"
        )

        print(
            f"Features            : {num_features}"
        )

        print(
            f"Qubits              : {num_features}"
        )

        print(
            f"Kernel pairs        : {len(pairs)}"
        )

        print(
            f"Pairs/process       : "
            f"{len(pairs) / size:.2f}"
        )

        print(
            f"Iterations          : {args.iterations}"
        )

        print(
            "Backend             : Qiskit CPU"
        )

        print(
            "Condition           : noiseless"
        )

        print()
        print(
            "All workload samples are distinct, "
            "non-overlapping real NinaPro "
            "512-sample EMG windows."
        )

        if num_samples != 1404:

            print()
            print(
                "WARNING: Expected 1404 windows "
                f"for S1_E1_A1.mat, found "
                f"{num_samples}."
            )

    # ---------------------------------------------------------
    # Quantum kernel
    # ---------------------------------------------------------

    kernel = QuantumKernel(
        reps=1,
        backend="cpu",
    )

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    comm.Barrier()

    warmup_matrix = (
        run_iteration(
            features,
            kernel,
            comm,
            pairs,
        )
    )

    comm.Barrier()

    if rank == 0:

        warmup_valid = (
            validate_matrix(
                warmup_matrix
            )
        )

        print()
        print(
            f"Warm-up: "
            f"valid={warmup_valid}"
        )

        if not warmup_valid:

            raise RuntimeError(
                "Warm-up kernel matrix "
                "failed validation."
            )

    # ---------------------------------------------------------
    # Timed iterations
    # ---------------------------------------------------------

    iteration_times = []

    for iteration in range(
        1,
        args.iterations + 1,
    ):

        comm.Barrier()

        start = (
            time.perf_counter()
        )

        matrix = (
            run_iteration(
                features,
                kernel,
                comm,
                pairs,
            )
        )

        comm.Barrier()

        elapsed = (
            time.perf_counter()
            - start
        )

        if rank == 0:

            valid = (
                validate_matrix(
                    matrix
                )
            )

            print(
                f"Iteration {iteration}: "
                f"{elapsed:.6f} s "
                f"(valid={valid})"
            )

            if not valid:

                raise RuntimeError(
                    "Timed kernel matrix "
                    "failed validation."
                )

            iteration_times.append(
                elapsed
            )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    if rank == 0:

        mean_time, std_time = (
            save_result(
                results_file=(
                    args.results_file
                ),
                processes=size,
                node_count=node_count,
                ranks_per_node=(
                    ranks_per_node
                ),
                features=features,
                pairs=pairs,
                iterations=(
                    args.iterations
                ),
                iteration_times=(
                    iteration_times
                ),
                windows_per_segment=(
                    windows_per_segment
                ),
            )
        )

        print()
        print("=" * 72)
        print("MPI SUMMARY")
        print("=" * 72)

        print(
            f"Processes     : {size}"
        )

        print(
            f"Nodes         : {node_count}"
        )

        print(
            f"Segments      : "
            f"{len(windows_per_segment)}"
        )

        print(
            f"Real windows  : {num_samples}"
        )

        print(
            f"Kernel pairs  : {len(pairs)}"
        )

        print(
            f"Pairs/rank    : "
            f"{len(pairs) / size:.2f}"
        )

        print(
            f"Mean time     : "
            f"{mean_time:.6f} s"
        )

        print(
            f"Std time      : "
            f"{std_time:.6f} s"
        )

        print()

        print(
            "Results saved to:"
        )

        print(
            args.results_file
        )


if __name__ == "__main__":
    main()