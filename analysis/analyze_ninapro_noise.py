"""
Analyze NinaPro real-data noise validation results.

Reads:
    results/logs/ninapro_noise_summary.csv

Creates:
    results/plots/ninapro_noise_fidelity.png
    results/plots/ninapro_noise_runtime.png
"""

from pathlib import Path
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "logs"
    / "ninapro_noise_summary.csv"
)

PLOTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "plots"
)


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------

def load_summary():

    if not SUMMARY_FILE.exists():

        raise FileNotFoundError(
            f"Summary file not found:\n{SUMMARY_FILE}"
        )

    rows = []

    with SUMMARY_FILE.open(
        "r",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            rows.append({
                "noise_strength": float(
                    row["noise_strength"]
                ),
                "mean_fidelity": float(
                    row["mean_fidelity"]
                ),
                "std_fidelity": float(
                    row["std_fidelity"]
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
            "No rows found in noise summary."
        )

    rows.sort(
        key=lambda row: row["noise_strength"]
    )

    return rows


# ------------------------------------------------------------
# Fidelity plot
# ------------------------------------------------------------

def plot_fidelity(rows):

    noise = np.asarray([
        row["noise_strength"]
        for row in rows
    ])

    fidelity = np.asarray([
        row["mean_fidelity"]
        for row in rows
    ])

    std = np.asarray([
        row["std_fidelity"]
        for row in rows
    ])

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.errorbar(
        noise,
        fidelity,
        yerr=std,
        marker="o",
        linewidth=2,
        capsize=4,
    )

    axis.set_title(
        "NinaPro Quantum-State Fidelity Under Depolarizing Noise"
    )

    axis.set_xlabel(
        "Depolarizing Noise Strength (p)"
    )

    axis.set_ylabel(
        "Mean Fidelity"
    )

    axis.set_ylim(
        max(0.0, np.min(fidelity - std) - 0.0001),
        1.0001,
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    output = (
        PLOTS_DIR
        / "ninapro_noise_fidelity.png"
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
# Runtime plot
# ------------------------------------------------------------

def plot_runtime(rows):

    noise = np.asarray([
        row["noise_strength"]
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
        noise,
        runtime,
        yerr=std,
        marker="o",
        linewidth=2,
        capsize=4,
    )

    axis.set_title(
        "NinaPro Noise Simulation Runtime"
    )

    axis.set_xlabel(
        "Depolarizing Noise Strength (p)"
    )

    axis.set_ylabel(
        "Mean Runtime per Sample (seconds)"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    output = (
        PLOTS_DIR
        / "ninapro_noise_runtime.png"
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
    print("NinaPro Noise Result Analysis")
    print("=" * 70)

    print(
        f"\nInput:\n{SUMMARY_FILE}"
    )

    rows = load_summary()

    print(
        f"\nLoaded {len(rows)} noise conditions."
    )

    for row in rows:

        print(
            f"p={row['noise_strength']:.3f}  "
            f"fidelity={row['mean_fidelity']:.8f}  "
            f"runtime={row['mean_runtime_seconds']:.6f} s"
        )

    print("\nGenerating plots...")
    print("-" * 70)

    plot_fidelity(rows)
    plot_runtime(rows)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()