"""
Analyze the final Bridges-2 multi-node MPI topology experiment.

Compares matched MPI rank counts using single-node and
multi-node placement for the fixed 102-sample NinaPro
quantum-kernel workload.

Outputs:
    results/logs/ninapro_mpi_multinode_summary.csv
    results/plots/ninapro_mpi_multinode_runtime.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIR = (
    PROJECT_ROOT
    / "results"
    / "logs"
)

PLOT_DIR = (
    PROJECT_ROOT
    / "results"
    / "plots"
)

SUMMARY_FILE = (
    LOG_DIR
    / "ninapro_mpi_multinode_summary.csv"
)

PLOT_FILE = (
    PLOT_DIR
    / "ninapro_mpi_multinode_runtime.png"
)


CONFIGURATIONS = [
    {
        "ranks": 8,
        "nodes": 1,
        "ranks_per_node": 8,
        "placement": "single_node",
        "file": "ninapro_mpi_multinode_8r_1n.csv",
    },
    {
        "ranks": 8,
        "nodes": 2,
        "ranks_per_node": 4,
        "placement": "multi_node",
        "file": "ninapro_mpi_multinode_8r_2n.csv",
    },
    {
        "ranks": 16,
        "nodes": 1,
        "ranks_per_node": 16,
        "placement": "single_node",
        "file": "ninapro_mpi_multinode_16r_1n.csv",
    },
    {
        "ranks": 16,
        "nodes": 2,
        "ranks_per_node": 8,
        "placement": "multi_node",
        "file": "ninapro_mpi_multinode_16r_2n.csv",
    },
    {
        "ranks": 32,
        "nodes": 1,
        "ranks_per_node": 32,
        "placement": "single_node",
        "file": "ninapro_mpi_multinode_32r_1n.csv",
    },
    {
        "ranks": 32,
        "nodes": 4,
        "ranks_per_node": 8,
        "placement": "multi_node",
        "file": "ninapro_mpi_multinode_32r_4n.csv",
    },
]


def main():

    rows = []

    for configuration in CONFIGURATIONS:

        path = (
            LOG_DIR
            / configuration["file"]
        )

        data = pd.read_csv(path)

        result = data.iloc[-1]

        rows.append(
            {
                "mpi_ranks": configuration["ranks"],
                "nodes": configuration["nodes"],
                "ranks_per_node": configuration["ranks_per_node"],
                "placement": configuration["placement"],
                "mean_runtime_seconds": result[
                    "mean_runtime_seconds"
                ],
                "std_runtime_seconds": result[
                    "std_runtime_seconds"
                ],
            }
        )

    summary = pd.DataFrame(rows)

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("=" * 70)
    print("NinaPro Multi-Node MPI Summary")
    print("=" * 70)
    print(summary.to_string(index=False))

    print()
    print("Matched-rank topology overhead")
    print("-" * 70)

    for ranks in [
        8,
        16,
        32,
    ]:

        single = summary[
            (
                summary["mpi_ranks"] == ranks
            )
            &
            (
                summary["placement"]
                == "single_node"
            )
        ].iloc[0]

        multi = summary[
            (
                summary["mpi_ranks"] == ranks
            )
            &
            (
                summary["placement"]
                == "multi_node"
            )
        ].iloc[0]

        overhead = (
            (
                multi["mean_runtime_seconds"]
                /
                single["mean_runtime_seconds"]
            )
            - 1.0
        ) * 100.0

        print(
            f"{ranks:2d} ranks: "
            f"{overhead:.2f}% "
            f"higher runtime with multi-node placement"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    ranks = np.array([
        8,
        16,
        32,
    ])

    single = (
        summary[
            summary["placement"]
            == "single_node"
        ]
        .sort_values("mpi_ranks")
    )

    multi = (
        summary[
            summary["placement"]
            == "multi_node"
        ]
        .sort_values("mpi_ranks")
    )

    x = np.arange(
        len(ranks)
    )

    width = 0.35

    fig, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.bar(
        x - width / 2,
        single["mean_runtime_seconds"],
        width,
        yerr=single["std_runtime_seconds"],
        capsize=4,
        label="Single node",
    )

    axis.bar(
        x + width / 2,
        multi["mean_runtime_seconds"],
        width,
        yerr=multi["std_runtime_seconds"],
        capsize=4,
        label="Multi-node",
    )

    axis.set_xlabel(
        "MPI ranks"
    )

    axis.set_ylabel(
        "Mean runtime (s)"
    )

    axis.set_title(
        "NinaPro MPI Topology Comparison"
    )

    axis.set_xticks(x)
    axis.set_xticklabels(
        [
            str(rank)
            for rank in ranks
        ]
    )

    axis.legend()

    fig.tight_layout()

    fig.savefig(
        PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(
        f"Summary saved to:\n"
        f"{SUMMARY_FILE}"
    )

    print(
        f"\nPlot saved to:\n"
        f"{PLOT_FILE}"
    )


if __name__ == "__main__":
    main()