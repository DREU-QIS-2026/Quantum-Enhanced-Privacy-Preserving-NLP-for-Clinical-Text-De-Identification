"""
QuantumHPC final project entry point.

The completed research project is made of several independent benchmark,
MPI/SLURM, and analysis workflows.

This command-line interface provides one organized front door without
rewriting or hiding the validated experiment scripts.

Examples
--------

    python main.py

    python main.py status

    python main.py benchmark --backend cpu

    python main.py noise

    python main.py qubit-scaling

    python main.py analyze fullcapacity

    python main.py analyze all

    python main.py hpc

    python main.py legacy --epsilon 1 --condition noiseless
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import config


# ============================================================
# General Helpers
# ============================================================

def print_header(title: str) -> None:

    print(
        "\n"
        + "=" * 72
    )

    print(title)

    print(
        "=" * 72
    )


def relative_path(
    path: Path,
) -> str:

    try:

        return str(
            path.relative_to(
                config.PROJECT_ROOT
            )
        )

    except ValueError:

        return str(path)


def run_python_script(
    script: Path,
    extra_args: list[str] | None = None,
) -> None:

    """
    Run one existing project script using the
    currently active Python interpreter.
    """

    if not script.exists():

        raise SystemExit(
            "Required script does not exist:\n"
            f"  {script}"
        )

    command = [
        sys.executable,
        str(script),
    ]

    if extra_args:

        command.extend(
            extra_args
        )

    print_header(
        "Running"
    )

    print(
        " ".join(command)
    )

    subprocess.run(
        command,
        cwd=config.PROJECT_ROOT,
        check=True,
    )


def print_file_status(
    label: str,
    path: Path,
) -> None:

    if path.exists():

        state = "OK"

    else:

        state = "MISSING"

    print(
        f"[{state:<7}] "
        f"{label:<34} "
        f"{relative_path(path)}"
    )


# ============================================================
# Project Overview
# ============================================================

def show_overview() -> None:

    print_header(
        "QuantumHPC"
    )

    print(
        "Final DREU-QIS HPC workflow for "
        "biomedical quantum-kernel simulation."
    )

    print(
        "\nPrimary final workload:"
    )

    print(
        "  NinaPro recording : "
        f"{relative_path(config.NINAPRO_DATASET_PATH)}"
    )

    print(
        "  EMG windows       : "
        f"{config.FULLCAPACITY_WINDOWS:,}"
    )

    print(
        "  Quantum features  : "
        f"{config.NUM_QUBITS}"
    )

    print(
        "  Kernel pairs      : "
        f"{config.FULLCAPACITY_KERNEL_PAIRS:,}"
    )

    print(
        "  Max MPI ranks     : 128"
    )

    print(
        "  Max Bridges nodes : 16"
    )

    print(
        "\nUseful commands:"
    )

    print(
        "  python main.py status"
    )

    print(
        "  python main.py benchmark --backend cpu"
    )

    print(
        "  python main.py noise"
    )

    print(
        "  python main.py qubit-scaling"
    )

    print(
        "  python main.py analyze all"
    )

    print(
        "  python main.py hpc"
    )

    print(
        "  python main.py legacy --help"
    )

    print(
        "\nRun 'python main.py --help' "
        "for the complete command list."
    )


# ============================================================
# Project Status
# ============================================================

def show_status() -> None:

    print_header(
        "QuantumHPC Project Status"
    )

    print(
        "\nDataset"
    )

    print(
        "-" * 72
    )

    print_file_status(
        "NinaPro S1_E1_A1.mat",
        config.NINAPRO_DATASET_PATH,
    )

    print(
        "\nBenchmark drivers"
    )

    print(
        "-" * 72
    )

    for (
        name,
        path,
    ) in config.BENCHMARK_SCRIPTS.items():

        print_file_status(
            name,
            path,
        )

    print(
        "\nAnalysis scripts"
    )

    print(
        "-" * 72
    )

    for (
        name,
        path,
    ) in config.ANALYSIS_SCRIPTS.items():

        print_file_status(
            name,
            path,
        )

    print(
        "\nSLURM workflows"
    )

    print(
        "-" * 72
    )

    for (
        name,
        path,
    ) in config.SLURM_SCRIPTS.items():

        print_file_status(
            name,
            path,
        )

    print(
        "\nSelected final result files"
    )

    print(
        "-" * 72
    )

    for (
        name,
        path,
    ) in config.FINAL_RESULT_FILES.items():

        print_file_status(
            name,
            path,
        )

    print(
        "\nSelected final plots"
    )

    print(
        "-" * 72
    )

    for (
        name,
        path,
    ) in config.FINAL_PLOT_FILES.items():

        print_file_status(
            name,
            path,
        )


# ============================================================
# Final Local Benchmark Workflows
# ============================================================

def run_local_benchmark(
    backend: str,
) -> None:

    run_python_script(
        config.LOCAL_BENCHMARK_SCRIPT,
        [
            "--backend",
            backend,
        ],
    )


def run_noise_benchmark() -> None:

    run_python_script(
        config.NOISE_BENCHMARK_SCRIPT
    )


def run_qubit_scaling() -> None:

    run_python_script(
        config.QUBIT_SCALING_SCRIPT
    )


# ============================================================
# Analysis Workflows
# ============================================================

def run_analysis(
    target: str,
) -> None:

    if target == "all":

        targets = list(
            config.ANALYSIS_SCRIPTS.keys()
        )

    else:

        targets = [
            target
        ]

    for name in targets:

        script = (
            config.ANALYSIS_SCRIPTS[
                name
            ]
        )

        run_python_script(
            script
        )


# ============================================================
# Bridges-2 / HPC Workflows
# ============================================================

def show_hpc_workflows() -> None:

    print_header(
        "Bridges-2 / HPC Workflows"
    )

    print(
        "Final MPI and Bridges-2 GPU experiments "
        "are intentionally kept as SLURM workflows "
        "rather than being launched automatically "
        "by main.py."
    )

    print(
        "\nThis preserves the exact resource "
        "requests and MPI rank placement used "
        "for the research results."
    )

    print(
        "\nAvailable SLURM files:"
    )

    for (
        name,
        path,
    ) in config.SLURM_SCRIPTS.items():

        if path.exists():

            state = "available"

        else:

            state = "missing"

        print(
            f"  {name:<28} "
            f"{relative_path(path)} "
            f"({state})"
        )

    print(
        "\nSubmit the required workflow "
        "directly with sbatch on Bridges-2."
    )

    print(
        "\nFor the final full-capacity experiment, "
        "topology-specific resource flags were "
        "supplied to sbatch so that total MPI "
        "ranks and physical-node placement could "
        "be controlled explicitly."
    )


# ============================================================
# Original / Legacy Framework
# ============================================================

def run_legacy_experiment(
    epsilon: float,
    condition: str,
) -> None:

    """
    Run the original sample-clinical
    ExperimentRunner workflow.
    """

    # Import lazily so normal final-project
    # commands do not depend on the historical
    # experiment framework.

    from experiment.experiment_config import (
        ExperimentConfig,
    )

    from experiment.experiment_runner import (
        ExperimentRunner,
    )

    print_header(
        "Legacy QuantumHPC ExperimentRunner"
    )

    print(
        "This command runs the original "
        "sample-clinical project framework."
    )

    print(
        "\nIt is retained for project history "
        "and compatibility; it is not the final "
        "NinaPro HPC benchmark pipeline."
    )

    experiment_config = ExperimentConfig(
        epsilon=epsilon,
        condition=condition,
    )

    runner = ExperimentRunner(
        experiment_config
    )

    runner.run()


# ============================================================
# Command-Line Interface
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "QuantumHPC final project entry point. "
            "Run local benchmarks, reproduce analysis, "
            "inspect final project state, or locate "
            "Bridges-2 workflows."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )


    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    subparsers.add_parser(
        "status",
        help=(
            "Show datasets, scripts, results, "
            "plots, and workflow status."
        ),
    )


    # --------------------------------------------------------
    # Local benchmark
    # --------------------------------------------------------

    benchmark_parser = (
        subparsers.add_parser(
            "benchmark",
            help=(
                "Run the final 102-sample local "
                "NinaPro CPU/GPU benchmark."
            ),
        )
    )

    benchmark_parser.add_argument(
        "--backend",
        choices=[
            "cpu",
            "gpu",
            "both",
        ],
        default="cpu",
        help=(
            "Simulation backend. "
            "'gpu' and 'both' require the "
            "local qsim CUDA environment."
        ),
    )


    # --------------------------------------------------------
    # Noise benchmark
    # --------------------------------------------------------

    subparsers.add_parser(
        "noise",
        help=(
            "Run the final NinaPro "
            "depolarizing-noise benchmark."
        ),
    )


    # --------------------------------------------------------
    # Qubit scaling
    # --------------------------------------------------------

    subparsers.add_parser(
        "qubit-scaling",
        help=(
            "Run the local CPU/GPU "
            "qubit-scaling benchmark."
        ),
    )


    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    analysis_parser = (
        subparsers.add_parser(
            "analyze",
            help=(
                "Regenerate final analysis "
                "tables and plots from saved results."
            ),
        )
    )

    analysis_parser.add_argument(
        "target",
        choices=[
            *config.ANALYSIS_SCRIPTS.keys(),
            "all",
        ],
        help=(
            "Analysis workflow to run."
        ),
    )


    # --------------------------------------------------------
    # HPC information
    # --------------------------------------------------------

    subparsers.add_parser(
        "hpc",
        help=(
            "List the final Bridges-2 "
            "SLURM workflows."
        ),
    )


    # --------------------------------------------------------
    # Legacy framework
    # --------------------------------------------------------

    legacy_parser = (
        subparsers.add_parser(
            "legacy",
            help=(
                "Run the original sample-clinical "
                "ExperimentRunner workflow."
            ),
        )
    )

    legacy_parser.add_argument(
        "--epsilon",
        type=float,
        default=config.DEFAULT_EPSILON,
        help=(
            "Legacy differential-privacy "
            "epsilon value."
        ),
    )

    legacy_parser.add_argument(
        "--condition",
        choices=config.NOISE_CONDITIONS,
        default=config.DEFAULT_CONDITION,
        help=(
            "Legacy simulation condition."
        ),
    )

    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    # Running main.py with no arguments gives
    # a project overview rather than launching
    # a potentially expensive experiment.

    if args.command is None:

        show_overview()

        return


    if args.command == "status":

        show_status()


    elif args.command == "benchmark":

        run_local_benchmark(
            args.backend
        )


    elif args.command == "noise":

        run_noise_benchmark()


    elif args.command == "qubit-scaling":

        run_qubit_scaling()


    elif args.command == "analyze":

        run_analysis(
            args.target
        )


    elif args.command == "hpc":

        show_hpc_workflows()


    elif args.command == "legacy":

        run_legacy_experiment(
            epsilon=args.epsilon,
            condition=args.condition,
        )


    else:

        parser.error(
            f"Unknown command: "
            f"{args.command}"
        )


if __name__ == "__main__":

    main()