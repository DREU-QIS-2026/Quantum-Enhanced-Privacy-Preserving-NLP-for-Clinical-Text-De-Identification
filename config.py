"""
Central configuration for the final QuantumHPC project.

The final project is experiment-oriented: individual benchmark and analysis
scripts remain independent so that local CPU/GPU, MPI, Bridges-2 GPU, and
post-processing workflows can be reproduced separately. main.py uses this
file as a lightweight project index and command-line entry point.

Legacy configuration values are retained at the bottom of this file so the
original ExperimentRunner and early MPI prototype remain runnable.
"""

from pathlib import Path
import os


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = PROJECT_ROOT / "datasets"
TESTS_DIR = PROJECT_ROOT / "tests"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
SLURM_DIR = PROJECT_ROOT / "slurm"
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULT_LOGS_DIR = RESULTS_DIR / "logs"
RESULT_PLOTS_DIR = RESULTS_DIR / "plots"
RESULT_TABLES_DIR = RESULTS_DIR / "tables"
RESULT_EXPORTS_DIR = RESULTS_DIR / "exports"

for directory in [
    RESULTS_DIR,
    RESULT_LOGS_DIR,
    RESULT_PLOTS_DIR,
    RESULT_TABLES_DIR,
    RESULT_EXPORTS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Final NinaPro Workload
# ============================================================

NINAPRO_DATASET_PATH = (
    DATA_DIR
    / "ninapro"
    / "DB2"
    / "s1"
    / "S1_E1_A1.mat"
)

WINDOW_SIZE = 512
NUM_EMG_CHANNELS = 12
CLASSICAL_FEATURES_PER_CHANNEL = 3
NUM_CLASSICAL_FEATURES = 36

NUM_QUBITS = 8
FEATURE_MAP_REPS = 1
DEFAULT_TIMED_RUNS = 5


# ------------------------------------------------------------
# Final real-data workload sizes
# ------------------------------------------------------------

PRIMARY_WINDOWS = 102
PRIMARY_KERNEL_PAIRS = 5_253

BALANCED_WINDOWS = 714
BALANCED_KERNEL_PAIRS = 255_255

FULLCAPACITY_WINDOWS = 1_404
FULLCAPACITY_KERNEL_PAIRS = 986_310


# ============================================================
# Final Benchmark Drivers
# ============================================================

LOCAL_BENCHMARK_SCRIPT = (
    TESTS_DIR
    / "ninapro_benchmark.py"
)

QUBIT_SCALING_SCRIPT = (
    TESTS_DIR
    / "qubit_scaling_test.py"
)

NOISE_BENCHMARK_SCRIPT = (
    TESTS_DIR
    / "ninapro_noise_benchmark.py"
)

MPI_BENCHMARK_SCRIPT = (
    TESTS_DIR
    / "ninapro_mpi_benchmark.py"
)

REALDATA_MPI_BENCHMARK_SCRIPT = (
    TESTS_DIR
    / "ninapro_mpi_realdata_benchmark.py"
)

FULLCAPACITY_MPI_BENCHMARK_SCRIPT = (
    TESTS_DIR
    / "ninapro_mpi_fullcapacity_benchmark.py"
)

BRIDGES_GPU_BENCHMARK_SCRIPT = (
    TESTS_DIR
    / "ninapro_bridges_gpu_benchmark.py"
)


BENCHMARK_SCRIPTS = {
    "local": LOCAL_BENCHMARK_SCRIPT,
    "qubit-scaling": QUBIT_SCALING_SCRIPT,
    "noise": NOISE_BENCHMARK_SCRIPT,
    "mpi": MPI_BENCHMARK_SCRIPT,
    "mpi-realdata": REALDATA_MPI_BENCHMARK_SCRIPT,
    "mpi-fullcapacity": FULLCAPACITY_MPI_BENCHMARK_SCRIPT,
    "bridges-gpu": BRIDGES_GPU_BENCHMARK_SCRIPT,
}


# ============================================================
# Final Analysis Scripts
# ============================================================

ANALYSIS_SCRIPTS = {
    "noise":
        ANALYSIS_DIR
        / "analyze_ninapro_noise.py",

    "mpi":
        ANALYSIS_DIR
        / "analyze_ninapro_mpi.py",

    "multinode":
        ANALYSIS_DIR
        / "analyze_ninapro_multinode.py",

    "gpu":
        ANALYSIS_DIR
        / "analyze_ninapro_bridges_gpu.py",

    "fullcapacity":
        ANALYSIS_DIR
        / "analyze_ninapro_fullcapacity_mpi.py",
}


# ============================================================
# Bridges-2 SLURM Workflows
# ============================================================

SLURM_SCRIPTS = {
    "mpi-final":
        SLURM_DIR
        / "ninapro_final_bridges.slurm",

    "mpi-multinode":
        SLURM_DIR
        / "ninapro_multinode_bridges.slurm",

    "mpi-realdata-smoke":
        SLURM_DIR
        / "ninapro_realdata_smoke.slurm",

    "mpi-realdata-final":
        SLURM_DIR
        / "ninapro_realdata_final_bridges.slurm",

    "mpi-fullcapacity-smoke":
        SLURM_DIR
        / "ninapro_fullcapacity_smoke.slurm",

    "mpi-fullcapacity-final":
        SLURM_DIR
        / "ninapro_fullcapacity_final_bridges.slurm",

    "gpu-final":
        SLURM_DIR
        / "ninapro_final_gpu_bridges.slurm",
}


# ============================================================
# Selected Final Result Files
# ============================================================

FINAL_RESULT_FILES = {
    "local CPU/GPU benchmark":
        RESULT_LOGS_DIR
        / "ninapro_benchmark.csv",

    "noise samples":
        RESULT_LOGS_DIR
        / "ninapro_noise_samples.csv",

    "noise summary":
        RESULT_LOGS_DIR
        / "ninapro_noise_summary.csv",

    "MPI strong scaling":
        RESULT_LOGS_DIR
        / "ninapro_mpi_strong_scaling.csv",

    "MPI weak scaling":
        RESULT_LOGS_DIR
        / "ninapro_mpi_weak_scaling.csv",

    "Bridges-2 GPU benchmark":
        RESULT_LOGS_DIR
        / "ninapro_bridges_gpu_benchmark.csv",

    "Bridges-2 GPU qubit scaling":
        RESULT_LOGS_DIR
        / "qubit_scaling_bridges_gpu.csv",

    "full-capacity MPI summary":
        RESULT_LOGS_DIR
        / "ninapro_mpi_fullcapacity_summary.csv",

    "full-capacity topology summary":
        RESULT_LOGS_DIR
        / "ninapro_mpi_fullcapacity_topology_summary.csv",
}


FINAL_PLOT_FILES = {
    "noise fidelity":
        RESULT_PLOTS_DIR
        / "ninapro_noise_fidelity.png",

    "noise runtime":
        RESULT_PLOTS_DIR
        / "ninapro_noise_runtime.png",

    "Bridges-2 backend runtime":
        RESULT_PLOTS_DIR
        / "ninapro_bridges_backend_runtime.png",

    "Bridges-2 GPU speedup":
        RESULT_PLOTS_DIR
        / "qubit_scaling_bridges_speedup.png",

    "full-capacity process scaling":
        RESULT_PLOTS_DIR
        / "ninapro_fullcapacity_process_scaling.png",

    "full-capacity node scaling":
        RESULT_PLOTS_DIR
        / "ninapro_fullcapacity_node_scaling.png",
}


# ============================================================
# Legacy Framework Compatibility
# ============================================================
#
# The original project framework used sample_clinical.csv,
# differential-privacy epsilon values, measurement counts, and
# the ExperimentRunner.
#
# These values are intentionally retained so that:
#
#     python main.py legacy
#
# can still execute the original framework and older project
# modules do not break.
#
# They are NOT the configuration used by the final NinaPro HPC
# benchmarking workflow.
# ============================================================

DATASET_NAME = "sample_clinical.csv"
DATASET_PATH = (
    DATA_DIR
    / DATASET_NAME
)

SHOTS = 1024
BACKEND = "cpu"

DEFAULT_EPSILON = 1.0
DELTA = 1e-5

DEFAULT_CONDITION = "noiseless"

EPSILON_VALUES = [
    1.0,
    2.0,
    4.0,
    8.0,
]

NOISE_CONDITIONS = [
    "noiseless",
    "depolarizing",
    "amplitude_damping",
]


# ------------------------------------------------------------
# Original framework output locations
# ------------------------------------------------------------

PLOTS_DIR = (
    PROJECT_ROOT
    / "plots"
)

FEATURE_MAP_DIR = (
    PLOTS_DIR
    / "feature_maps"
)

CIRCUIT_DIR = (
    PLOTS_DIR
    / "circuits"
)

RUNTIME_DIR = (
    PLOTS_DIR
    / "runtime"
)

THROUGHPUT_DIR = (
    PLOTS_DIR
    / "throughput"
)

SCALING_DIR = (
    PLOTS_DIR
    / "scaling"
)

COMPARISON_DIR = (
    PLOTS_DIR
    / "comparisons"
)


TABLES_DIR = RESULT_TABLES_DIR
EXPORTS_DIR = RESULT_EXPORTS_DIR


for directory in [
    PLOTS_DIR,
    FEATURE_MAP_DIR,
    CIRCUIT_DIR,
    RUNTIME_DIR,
    THROUGHPUT_DIR,
    SCALING_DIR,
    COMPARISON_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


RESULTS_FILE = Path(
    os.environ.get(
        "QUANTUMHPC_RESULTS_FILE",
        RESULT_LOGS_DIR
        / "results.csv",
    )
)