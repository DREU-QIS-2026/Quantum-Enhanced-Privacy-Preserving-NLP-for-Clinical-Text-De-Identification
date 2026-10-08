"""
Analyze the final Bridges-2 CPU/V100 benchmark.

Inputs:
    results/logs/ninapro_bridges_gpu_benchmark.csv
    results/logs/qubit_scaling_bridges_gpu.csv

Outputs:
    results/logs/ninapro_bridges_gpu_summary.csv
    results/logs/qubit_scaling_bridges_gpu_summary.csv
    results/logs/qubit_scaling_bridges_gpu_speedup.csv

    results/plots/ninapro_bridges_backend_runtime.png
    results/plots/qubit_scaling_bridges_runtime.png
    results/plots/qubit_scaling_bridges_speedup.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = PROJECT_ROOT / "results" / "logs"
PLOT_DIR = PROJECT_ROOT / "results" / "plots"

NINAPRO_FILE = (
    LOG_DIR
    / "ninapro_bridges_gpu_benchmark.csv"
)

SCALING_FILE = (
    LOG_DIR
    / "qubit_scaling_bridges_gpu.csv"
)

NINAPRO_SUMMARY_FILE = (
    LOG_DIR
    / "ninapro_bridges_gpu_summary.csv"
)

SCALING_SUMMARY_FILE = (
    LOG_DIR
    / "qubit_scaling_bridges_gpu_summary.csv"
)

SPEEDUP_FILE = (
    LOG_DIR
    / "qubit_scaling_bridges_gpu_speedup.csv"
)


EXPECTED_BACKENDS = [
    "cpu_statevector",
    "gpu_statevector",
    "gpu_tensor_network",
]

EXPECTED_QUBITS = [
    4,
    6,
    8,
    10,
    12,
    14,
    16,
]

EXPECTED_RUNS = 5


def summarize(data, group_columns):

    summary = (
        data
        .groupby(group_columns)["runtime_seconds"]
        .agg(
            runs="count",
            mean_seconds="mean",
            std_seconds="std",
            median_seconds="median",
            min_seconds="min",
            max_seconds="max",
        )
        .reset_index()
    )

    return summary


def validate_final_data(
    ninapro,
    scaling,
):

    if not ninapro["valid"].all():
        raise RuntimeError(
            "At least one NinaPro kernel "
            "failed validation."
        )

    if not scaling["valid"].all():
        raise RuntimeError(
            "At least one scaling kernel "
            "failed validation."
        )

    ninapro_backends = set(
        ninapro["backend"]
    )

    scaling_backends = set(
        scaling["backend"]
    )

    expected = set(
        EXPECTED_BACKENDS
    )

    if ninapro_backends != expected:
        raise RuntimeError(
            "Unexpected NinaPro backends: "
            f"{ninapro_backends}"
        )

    if scaling_backends != expected:
        raise RuntimeError(
            "Unexpected scaling backends: "
            f"{scaling_backends}"
        )

    if len(ninapro) != (
        len(EXPECTED_BACKENDS)
        * EXPECTED_RUNS
    ):
        raise RuntimeError(
            "Expected 15 final NinaPro rows, "
            f"found {len(ninapro)}."
        )

    expected_scaling_rows = (
        len(EXPECTED_QUBITS)
        * len(EXPECTED_BACKENDS)
        * EXPECTED_RUNS
    )

    if len(scaling) != expected_scaling_rows:
        raise RuntimeError(
            "Expected "
            f"{expected_scaling_rows} "
            "final scaling rows, "
            f"found {len(scaling)}."
        )

    actual_qubits = sorted(
        scaling["qubits"]
        .unique()
        .tolist()
    )

    if actual_qubits != EXPECTED_QUBITS:
        raise RuntimeError(
            "Unexpected qubit counts: "
            f"{actual_qubits}"
        )


def main():

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ninapro = pd.read_csv(
        NINAPRO_FILE
    )

    scaling = pd.read_csv(
        SCALING_FILE
    )

    validate_final_data(
        ninapro,
        scaling,
    )

    # --------------------------------------------------------
    # Full NinaPro workload summary
    # --------------------------------------------------------

    ninapro_summary = summarize(
        ninapro,
        ["backend"],
    )

    cpu_mean = float(
        ninapro_summary.loc[
            ninapro_summary["backend"]
            == "cpu_statevector",
            "mean_seconds",
        ].iloc[0]
    )

    ninapro_summary[
        "speedup_vs_cpu"
    ] = (
        cpu_mean
        / ninapro_summary[
            "mean_seconds"
        ]
    )

    ninapro_summary.to_csv(
        NINAPRO_SUMMARY_FILE,
        index=False,
    )

    print("=" * 72)
    print("Final NinaPro Bridges-2 Summary")
    print("=" * 72)
    print(
        ninapro_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Qubit scaling summary
    # --------------------------------------------------------

    scaling_summary = summarize(
        scaling,
        [
            "qubits",
            "backend",
        ],
    )

    scaling_summary.to_csv(
        SCALING_SUMMARY_FILE,
        index=False,
    )

    speedup_rows = []

    for qubits in EXPECTED_QUBITS:

        subset = scaling_summary[
            scaling_summary["qubits"]
            == qubits
        ]

        cpu = float(
            subset.loc[
                subset["backend"]
                == "cpu_statevector",
                "mean_seconds",
            ].iloc[0]
        )

        for backend in [
            "gpu_statevector",
            "gpu_tensor_network",
        ]:

            gpu = float(
                subset.loc[
                    subset["backend"]
                    == backend,
                    "mean_seconds",
                ].iloc[0]
            )

            speedup_rows.append(
                {
                    "qubits": qubits,
                    "backend": backend,
                    "cpu_mean_seconds": cpu,
                    "accelerator_mean_seconds": gpu,
                    "speedup_vs_cpu": (
                        cpu / gpu
                    ),
                }
            )

    speedups = pd.DataFrame(
        speedup_rows
    )

    speedups.to_csv(
        SPEEDUP_FILE,
        index=False,
    )

    print()
    print("=" * 72)
    print("Qubit Scaling Summary")
    print("=" * 72)
    print(
        scaling_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("GPU Speedup Relative to CPU")
    print("=" * 72)
    print(
        speedups.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Plot 1: complete NinaPro workload
    # --------------------------------------------------------

    plot_data = (
        ninapro_summary
        .set_index("backend")
        .loc[EXPECTED_BACKENDS]
        .reset_index()
    )

    fig, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.bar(
        plot_data["backend"],
        plot_data["mean_seconds"],
        yerr=plot_data["std_seconds"],
        capsize=4,
    )

    axis.set_xlabel(
        "Bridges-2 backend"
    )

    axis.set_ylabel(
        "Mean runtime (s)"
    )

    axis.set_title(
        "NinaPro 102-Sample Kernel on Bridges-2"
    )

    axis.tick_params(
        axis="x",
        rotation=15,
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR
        / "ninapro_bridges_backend_runtime.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Plot 2: qubit-scaling runtime
    # --------------------------------------------------------

    fig, axis = plt.subplots(
        figsize=(8, 5)
    )

    for backend in EXPECTED_BACKENDS:

        subset = (
            scaling_summary[
                scaling_summary["backend"]
                == backend
            ]
            .sort_values("qubits")
        )

        axis.errorbar(
            subset["qubits"],
            subset["mean_seconds"],
            yerr=subset["std_seconds"],
            marker="o",
            capsize=3,
            label=backend,
        )

    axis.set_xlabel(
        "Qubits"
    )

    axis.set_ylabel(
        "Mean runtime (s)"
    )

    axis.set_title(
        "Bridges-2 Quantum-Simulation Scaling"
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR
        / "qubit_scaling_bridges_runtime.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Plot 3: accelerator speedup
    # --------------------------------------------------------

    fig, axis = plt.subplots(
        figsize=(8, 5)
    )

    for backend in [
        "gpu_statevector",
        "gpu_tensor_network",
    ]:

        subset = (
            speedups[
                speedups["backend"]
                == backend
            ]
            .sort_values("qubits")
        )

        axis.plot(
            subset["qubits"],
            subset["speedup_vs_cpu"],
            marker="o",
            label=backend,
        )

    axis.axhline(
        1.0,
        linestyle="--",
    )

    axis.set_xlabel(
        "Qubits"
    )

    axis.set_ylabel(
        "Speedup relative to Aer CPU"
    )

    axis.set_title(
        "Bridges-2 V100 Speedup"
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR
        / "qubit_scaling_bridges_speedup.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("Saved:")
    print(NINAPRO_SUMMARY_FILE)
    print(SCALING_SUMMARY_FILE)
    print(SPEEDUP_FILE)

    print(
        PLOT_DIR
        / "ninapro_bridges_backend_runtime.png"
    )

    print(
        PLOT_DIR
        / "qubit_scaling_bridges_runtime.png"
    )

    print(
        PLOT_DIR
        / "qubit_scaling_bridges_speedup.png"
    )


if __name__ == "__main__":
    main()