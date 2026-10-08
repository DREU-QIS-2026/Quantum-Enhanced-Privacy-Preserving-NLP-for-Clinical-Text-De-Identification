"""
Analyze NinaPro MPI strong- and weak-scaling results.

Reads:

    results/logs/ninapro_mpi_strong_scaling.csv
    results/logs/ninapro_mpi_weak_scaling.csv

Creates:

    results/plots/ninapro_mpi_strong_runtime.png
    results/plots/ninapro_mpi_strong_speedup.png
    results/plots/ninapro_mpi_strong_efficiency.png
    results/plots/ninapro_mpi_weak_runtime.png
    results/plots/ninapro_mpi_weak_efficiency.png
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STRONG_FILE = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "ninapro_mpi_strong_scaling.csv"
)

WEAK_FILE = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "ninapro_mpi_weak_scaling.csv"
)

PLOTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "plots"
)


# ------------------------------------------------------------
# CSV loading
# ------------------------------------------------------------

def load_results(path):
    """
    Load an MPI scaling CSV and return numeric rows.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Results file not found:\n{path}"
        )

    rows = []

    with path.open(
        "r",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            rows.append({
                "processes": int(
                    row["processes"]
                ),
                "workload_samples": int(
                    row["workload_samples"]
                ),
                "kernel_pairs": int(
                    row["kernel_pairs"]
                ),
                "pairs_per_process": float(
                    row["pairs_per_process"]
                ),
                "mean_runtime_seconds": float(
                    row["mean_runtime_seconds"]
                ),
                "std_runtime_seconds": float(
                    row["std_runtime_seconds"]
                ),
            })

    if not rows:

        raise RuntimeError(
            f"No rows found in:\n{path}"
        )

    rows.sort(
        key=lambda row: row["processes"]
    )

    return rows


# ------------------------------------------------------------
# Strong scaling calculations
# ------------------------------------------------------------

def calculate_strong_scaling(rows):

    baseline = rows[0][
        "mean_runtime_seconds"
    ]

    for row in rows:

        runtime = row[
            "mean_runtime_seconds"
        ]

        speedup = (
            baseline / runtime
        )

        efficiency = (
            speedup
            / row["processes"]
        )

        row["speedup"] = speedup
        row["efficiency"] = efficiency

    return rows


# ------------------------------------------------------------
# Weak scaling calculations
# ------------------------------------------------------------

def calculate_weak_scaling(rows):

    baseline = rows[0][
        "mean_runtime_seconds"
    ]

    baseline_pairs_per_process = rows[0][
        "pairs_per_process"
    ]

    for row in rows:

        runtime = row[
            "mean_runtime_seconds"
        ]

        work_ratio = (
            row["pairs_per_process"]
            / baseline_pairs_per_process
        )

        efficiency = (
            baseline
            * work_ratio
            / runtime
        )

        row["weak_efficiency"] = (
            efficiency
        )

    return rows


# ------------------------------------------------------------
# Strong runtime
# ------------------------------------------------------------

def plot_strong_runtime(rows):

    processes = np.asarray([
        row["processes"]
        for row in rows
    ])

    runtime = np.asarray([
        row["mean_runtime_seconds"]
        for row in rows
    ])

    std = np.asarray([
        row["std_runtime_seconds"]
        for row in rows
    ])

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.errorbar(
        processes,
        runtime,
        yerr=std,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Measured runtime",
    )

    axis.set_title(
        "NinaPro MPI Strong Scaling — Runtime"
    )

    axis.set_xlabel(
        "MPI Processes"
    )

    axis.set_ylabel(
        "Mean Runtime (seconds)"
    )

    axis.set_xticks(
        processes
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    output = (
        PLOTS_DIR
        / "ninapro_mpi_strong_runtime.png"
    )

    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved:\n{output}"
    )


# ------------------------------------------------------------
# Strong speedup
# ------------------------------------------------------------

def plot_strong_speedup(rows):

    processes = np.asarray([
        row["processes"]
        for row in rows
    ])

    speedup = np.asarray([
        row["speedup"]
        for row in rows
    ])

    ideal = processes.astype(
        float
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.plot(
        processes,
        speedup,
        marker="o",
        linewidth=2,
        label="Measured speedup",
    )

    axis.plot(
        processes,
        ideal,
        linestyle="--",
        linewidth=2,
        label="Ideal linear speedup",
    )

    axis.set_title(
        "NinaPro MPI Strong Scaling — Speedup"
    )

    axis.set_xlabel(
        "MPI Processes"
    )

    axis.set_ylabel(
        "Speedup"
    )

    axis.set_xticks(
        processes
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    output = (
        PLOTS_DIR
        / "ninapro_mpi_strong_speedup.png"
    )

    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved:\n{output}"
    )


# ------------------------------------------------------------
# Strong efficiency
# ------------------------------------------------------------

def plot_strong_efficiency(rows):

    processes = np.asarray([
        row["processes"]
        for row in rows
    ])

    efficiency = np.asarray([
        row["efficiency"] * 100.0
        for row in rows
    ])

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.plot(
        processes,
        efficiency,
        marker="o",
        linewidth=2,
    )

    axis.axhline(
        100.0,
        linestyle="--",
        linewidth=2,
    )

    axis.set_title(
        "NinaPro MPI Strong Scaling — Parallel Efficiency"
    )

    axis.set_xlabel(
        "MPI Processes"
    )

    axis.set_ylabel(
        "Parallel Efficiency (%)"
    )

    axis.set_xticks(
        processes
    )

    axis.set_ylim(
        0,
        105,
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    output = (
        PLOTS_DIR
        / "ninapro_mpi_strong_efficiency.png"
    )

    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved:\n{output}"
    )


# ------------------------------------------------------------
# Weak runtime
# ------------------------------------------------------------

def plot_weak_runtime(rows):

    processes = np.asarray([
        row["processes"]
        for row in rows
    ])

    runtime = np.asarray([
        row["mean_runtime_seconds"]
        for row in rows
    ])

    std = np.asarray([
        row["std_runtime_seconds"]
        for row in rows
    ])

    baseline = runtime[0]

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.errorbar(
        processes,
        runtime,
        yerr=std,
        marker="o",
        linewidth=2,
        capsize=4,
        label="Measured runtime",
    )

    axis.axhline(
        baseline,
        linestyle="--",
        linewidth=2,
        label="Ideal constant runtime",
    )

    axis.set_title(
        "NinaPro MPI Weak Scaling — Runtime"
    )

    axis.set_xlabel(
        "MPI Processes"
    )

    axis.set_ylabel(
        "Mean Runtime (seconds)"
    )

    axis.set_xticks(
        processes
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    output = (
        PLOTS_DIR
        / "ninapro_mpi_weak_runtime.png"
    )

    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved:\n{output}"
    )


# ------------------------------------------------------------
# Weak efficiency
# ------------------------------------------------------------

def plot_weak_efficiency(rows):

    processes = np.asarray([
        row["processes"]
        for row in rows
    ])

    efficiency = np.asarray([
        row["weak_efficiency"] * 100.0
        for row in rows
    ])

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.plot(
        processes,
        efficiency,
        marker="o",
        linewidth=2,
    )

    axis.axhline(
        100.0,
        linestyle="--",
        linewidth=2,
    )

    axis.set_title(
        "NinaPro MPI Weak Scaling — Efficiency"
    )

    axis.set_xlabel(
        "MPI Processes"
    )

    axis.set_ylabel(
        "Weak-Scaling Efficiency (%)"
    )

    axis.set_xticks(
        processes
    )

    axis.set_ylim(
        0,
        105,
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    output = (
        PLOTS_DIR
        / "ninapro_mpi_weak_efficiency.png"
    )

    figure.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved:\n{output}"
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("NinaPro MPI Scaling Analysis")
    print("=" * 70)

    # --------------------------------------------------------
    # Strong scaling
    # --------------------------------------------------------

    print(
        f"\nLoading strong-scaling results:\n"
        f"{STRONG_FILE}"
    )

    strong_rows = load_results(
        STRONG_FILE
    )

    strong_rows = calculate_strong_scaling(
        strong_rows
    )

    print(
        f"\nStrong-scaling measurements: "
        f"{len(strong_rows)}"
    )

    for row in strong_rows:

        print(
            f"{row['processes']} processes  "
            f"runtime={row['mean_runtime_seconds']:.6f} s  "
            f"speedup={row['speedup']:.4f}  "
            f"efficiency={row['efficiency'] * 100:.2f}%"
        )

    print("\nGenerating strong-scaling plots...")
    print("-" * 70)

    plot_strong_runtime(
        strong_rows
    )

    plot_strong_speedup(
        strong_rows
    )

    plot_strong_efficiency(
        strong_rows
    )

    # --------------------------------------------------------
    # Weak scaling
    # --------------------------------------------------------

    print(
        f"\nLoading weak-scaling results:\n"
        f"{WEAK_FILE}"
    )

    weak_rows = load_results(
        WEAK_FILE
    )

    weak_rows = calculate_weak_scaling(
        weak_rows
    )

    print(
        f"\nWeak-scaling measurements: "
        f"{len(weak_rows)}"
    )

    for row in weak_rows:

        print(
            f"{row['processes']} processes  "
            f"samples={row['workload_samples']}  "
            f"pairs={row['kernel_pairs']}  "
            f"pairs/rank={row['pairs_per_process']:.2f}  "
            f"runtime={row['mean_runtime_seconds']:.6f} s  "
            f"efficiency={row['weak_efficiency'] * 100:.2f}%"
        )

    print("\nGenerating weak-scaling plots...")
    print("-" * 70)

    plot_weak_runtime(
        weak_rows
    )

    plot_weak_efficiency(
        weak_rows
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()