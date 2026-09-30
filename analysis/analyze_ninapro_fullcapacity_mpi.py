"""
Final analysis for the full-capacity NinaPro MPI experiment.

Workload:
    1404 distinct non-overlapping real NinaPro EMG windows
    986310 upper-triangular quantum-kernel pairs
    8 quantum features / qubits

Experiments:
    Single-node process scaling:
        1, 8, 16, 32, 64, 128 MPI ranks

    Fixed-128-rank physical-node scaling:
        1, 2, 4, 8, 16 Bridges-2 nodes

Outputs:
    results/logs/ninapro_mpi_fullcapacity_summary.csv
    results/logs/ninapro_mpi_fullcapacity_topology_summary.csv

    results/plots/ninapro_fullcapacity_process_scaling.png
    results/plots/ninapro_fullcapacity_node_scaling.png
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = PROJECT_ROOT / "results" / "logs"
PLOT_DIR = PROJECT_ROOT / "results" / "plots"

SUMMARY_FILE = (
    LOG_DIR
    / "ninapro_mpi_fullcapacity_summary.csv"
)

TOPOLOGY_FILE = (
    LOG_DIR
    / "ninapro_mpi_fullcapacity_topology_summary.csv"
)


# ---------------------------------------------------------------------
# Final configurations
# ---------------------------------------------------------------------

CONFIGURATIONS = [
    ("1r_1n", 1, 1),
    ("8r_1n", 8, 1),
    ("16r_1n", 16, 1),
    ("32r_1n", 32, 1),
    ("64r_1n", 64, 1),
    ("128r_1n", 128, 1),
    ("128r_2n", 128, 2),
    ("128r_4n", 128, 4),
    ("128r_8n", 128, 8),
    ("128r_16n", 128, 16),
]

EXPECTED_WINDOWS = 1404
EXPECTED_PAIRS = 986310
EXPECTED_SEGMENTS = 102
EXPECTED_FEATURES = 8
EXPECTED_QUBITS = 8
EXPECTED_ITERATIONS = 5


# ---------------------------------------------------------------------
# Load and validate
# ---------------------------------------------------------------------

def load_results():

    rows = []

    for tag, expected_ranks, expected_nodes in CONFIGURATIONS:

        path = (
            LOG_DIR
            / f"ninapro_mpi_fullcapacity_{tag}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing final result file: {path}"
            )

        data = pd.read_csv(path)

        if len(data) != 1:
            raise RuntimeError(
                f"{path.name} should contain exactly "
                f"one final result row, found {len(data)}."
            )

        row = data.iloc[0].copy()

        # -------------------------------------------------------------
        # Configuration validation
        # -------------------------------------------------------------

        if int(row["processes"]) != expected_ranks:
            raise RuntimeError(
                f"{tag}: expected {expected_ranks} ranks, "
                f"found {row['processes']}."
            )

        if int(row["nodes"]) != expected_nodes:
            raise RuntimeError(
                f"{tag}: expected {expected_nodes} nodes, "
                f"found {row['nodes']}."
            )

        if int(row["segments"]) != EXPECTED_SEGMENTS:
            raise RuntimeError(
                f"{tag}: unexpected segment count."
            )

        if int(row["real_windows"]) != EXPECTED_WINDOWS:
            raise RuntimeError(
                f"{tag}: expected {EXPECTED_WINDOWS} windows, "
                f"found {row['real_windows']}."
            )

        if int(row["features"]) != EXPECTED_FEATURES:
            raise RuntimeError(
                f"{tag}: unexpected feature count."
            )

        if int(row["qubits"]) != EXPECTED_QUBITS:
            raise RuntimeError(
                f"{tag}: unexpected qubit count."
            )

        if int(row["kernel_pairs"]) != EXPECTED_PAIRS:
            raise RuntimeError(
                f"{tag}: expected {EXPECTED_PAIRS} pairs, "
                f"found {row['kernel_pairs']}."
            )

        if int(row["iterations"]) != EXPECTED_ITERATIONS:
            raise RuntimeError(
                f"{tag}: expected {EXPECTED_ITERATIONS} iterations, "
                f"found {row['iterations']}."
            )

        if str(row["condition"]).lower() != "noiseless":
            raise RuntimeError(
                f"{tag}: unexpected simulation condition."
            )

        if str(row["backend"]).lower() != "cpu":
            raise RuntimeError(
                f"{tag}: unexpected backend."
            )

        rows.append(
            {
                "configuration": tag,
                "mpi_ranks": int(row["processes"]),
                "nodes": int(row["nodes"]),
                "ranks_per_node": float(
                    row["ranks_per_node"]
                ),
                "segments": int(row["segments"]),
                "min_windows_per_segment": int(
                    row["min_windows_per_segment"]
                ),
                "max_windows_per_segment": int(
                    row["max_windows_per_segment"]
                ),
                "mean_windows_per_segment": float(
                    row["mean_windows_per_segment"]
                ),
                "real_windows": int(
                    row["real_windows"]
                ),
                "features": int(row["features"]),
                "qubits": int(row["qubits"]),
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
                "min_runtime_seconds": float(
                    row["min_runtime_seconds"]
                ),
                "max_runtime_seconds": float(
                    row["max_runtime_seconds"]
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------

def calculate_metrics(results):

    baseline_runtime = float(
        results.loc[
            results["configuration"] == "1r_1n",
            "mean_runtime_seconds",
        ].iloc[0]
    )

    results["speedup_vs_1rank"] = (
        baseline_runtime
        / results["mean_runtime_seconds"]
    )

    results["parallel_efficiency_percent"] = (
        results["speedup_vs_1rank"]
        / results["mpi_ranks"]
        * 100.0
    )

    one_node_128_runtime = float(
        results.loc[
            results["configuration"] == "128r_1n",
            "mean_runtime_seconds",
        ].iloc[0]
    )

    results[
        "change_vs_128r_1n_percent"
    ] = float("nan")

    mask = results["mpi_ranks"] == 128

    results.loc[
        mask,
        "change_vs_128r_1n_percent",
    ] = (
        (
            results.loc[
                mask,
                "mean_runtime_seconds",
            ]
            / one_node_128_runtime
        )
        - 1.0
    ) * 100.0

    return results


# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

def print_summary(results):

    print("=" * 108)
    print("NinaPro Full-Capacity MPI Final Analysis")
    print("=" * 108)

    print()
    print(
        "Workload: "
        "1404 distinct real EMG windows, "
        "986310 kernel pairs"
    )

    print()

    process_data = results[
        results["nodes"] == 1
    ].copy()

    print("=" * 84)
    print("Single-Node Process Scaling")
    print("=" * 84)

    print(
        process_data[
            [
                "mpi_ranks",
                "nodes",
                "mean_runtime_seconds",
                "std_runtime_seconds",
                "speedup_vs_1rank",
                "parallel_efficiency_percent",
            ]
        ].to_string(
            index=False,
            formatters={
                "mean_runtime_seconds":
                    "{:.6f}".format,
                "std_runtime_seconds":
                    "{:.6f}".format,
                "speedup_vs_1rank":
                    "{:.3f}".format,
                "parallel_efficiency_percent":
                    "{:.2f}".format,
            },
        )
    )

    print()

    topology = results[
        results["mpi_ranks"] == 128
    ].copy()

    print("=" * 84)
    print("Fixed-128-Rank Physical-Node Scaling")
    print("=" * 84)

    print(
        topology[
            [
                "nodes",
                "ranks_per_node",
                "mean_runtime_seconds",
                "std_runtime_seconds",
                "speedup_vs_1rank",
                "change_vs_128r_1n_percent",
            ]
        ].to_string(
            index=False,
            formatters={
                "ranks_per_node":
                    "{:.1f}".format,
                "mean_runtime_seconds":
                    "{:.6f}".format,
                "std_runtime_seconds":
                    "{:.6f}".format,
                "speedup_vs_1rank":
                    "{:.3f}".format,
                "change_vs_128r_1n_percent":
                    "{:+.2f}".format,
            },
        )
    )


# ---------------------------------------------------------------------
# Plot 1: process-count scaling
# ---------------------------------------------------------------------

def plot_process_scaling(results):

    data = (
        results[
            results["nodes"] == 1
        ]
        .sort_values("mpi_ranks")
        .copy()
    )

    fig, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.errorbar(
        data["mpi_ranks"],
        data["mean_runtime_seconds"],
        yerr=data["std_runtime_seconds"],
        marker="o",
        capsize=4,
    )

    axis.set_xscale(
        "log",
        base=2,
    )

    axis.set_xticks(
        data["mpi_ranks"]
    )

    axis.set_xticklabels(
        [
            str(value)
            for value in data["mpi_ranks"]
        ]
    )

    axis.set_xlabel(
        "MPI processes"
    )

    axis.set_ylabel(
        "Mean runtime (s)"
    )

    axis.set_title(
        "Full-Capacity NinaPro MPI Process Scaling"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    output = (
        PLOT_DIR
        / "ninapro_fullcapacity_process_scaling.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ---------------------------------------------------------------------
# Plot 2: fixed-rank physical-node scaling
# ---------------------------------------------------------------------

def plot_node_scaling(results):

    data = (
        results[
            results["mpi_ranks"] == 128
        ]
        .sort_values("nodes")
        .copy()
    )

    one_node_runtime = float(
        data.loc[
            data["nodes"] == 1,
            "mean_runtime_seconds",
        ].iloc[0]
    )

    fig, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.errorbar(
        data["nodes"],
        data["mean_runtime_seconds"],
        yerr=data["std_runtime_seconds"],
        marker="o",
        capsize=4,
        label="128 MPI processes",
    )

    axis.axhline(
        one_node_runtime,
        linestyle="--",
        label="One-node runtime",
    )

    axis.set_xscale(
        "log",
        base=2,
    )

    axis.set_xticks(
        data["nodes"]
    )

    axis.set_xticklabels(
        [
            str(value)
            for value in data["nodes"]
        ]
    )

    axis.set_xlabel(
        "Physical Bridges-2 nodes"
    )

    axis.set_ylabel(
        "Mean runtime (s)"
    )

    axis.set_title(
        "Fixed 128-Rank Physical-Node Scaling"
    )

    axis.grid(
        True,
        alpha=0.25,
    )

    axis.legend()

    fig.tight_layout()

    output = (
        PLOT_DIR
        / "ninapro_fullcapacity_node_scaling.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_results()

    results = calculate_metrics(
        results
    )

    # ---------------------------------------------------------
    # Save complete final summary
    # ---------------------------------------------------------

    results.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Save fixed-128-rank topology summary
    # ---------------------------------------------------------

    topology = (
        results[
            results["mpi_ranks"] == 128
        ]
        .sort_values("nodes")
        .copy()
    )

    topology.to_csv(
        TOPOLOGY_FILE,
        index=False,
    )

    print_summary(
        results
    )

    process_plot = (
        plot_process_scaling(
            results
        )
    )

    node_plot = (
        plot_node_scaling(
            results
        )
    )

    print()
    print("=" * 84)
    print("Saved")
    print("=" * 84)

    print(SUMMARY_FILE)
    print(TOPOLOGY_FILE)
    print(process_plot)
    print(node_plot)


if __name__ == "__main__":
    main()