#!/bin/bash -l

#SBATCH --job-name=ninapro-realdata
#SBATCH --account=cis260310p
#SBATCH --partition=RM
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/ninapro-realdata-%j.out

set -e

cd ~/QuantumHPC

mkdir -p results/logs


# ------------------------------------------------------------
# Result tag supplied as first script argument
# ------------------------------------------------------------

RESULT_TAG="$1"

if [ -z "$RESULT_TAG" ]; then
    echo "ERROR: Missing result tag."
    echo "Example:"
    echo "sbatch ... slurm/ninapro_realdata_final_bridges.slurm 8r_1n"
    exit 1
fi

RESULTS_FILE="results/logs/ninapro_mpi_realdata_${RESULT_TAG}.csv"

rm -f "$RESULTS_FILE"


# ------------------------------------------------------------
# Prevent hidden CPU threading
# ------------------------------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1


# ------------------------------------------------------------
# Load environment
# ------------------------------------------------------------

module load openmpi/4.0.5-nvhpc21.2
module load anaconda3/2024.10-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate quantumhpc


# ------------------------------------------------------------
# Topology
# ------------------------------------------------------------

RANKS_PER_NODE=$(( SLURM_NTASKS / SLURM_JOB_NUM_NODES ))


echo "============================================================"
echo "NinaPro 714-Window Real-Data MPI Benchmark"
echo "============================================================"
echo
echo "Job ID          : $SLURM_JOB_ID"
echo "Result tag      : $RESULT_TAG"
echo "Node count      : $SLURM_JOB_NUM_NODES"
echo "Total MPI ranks : $SLURM_NTASKS"
echo "Ranks per node  : $RANKS_PER_NODE"
echo "Node list       : $SLURM_JOB_NODELIST"
echo "Results file    : $RESULTS_FILE"
echo

echo "Allocated nodes:"
scontrol show hostnames "$SLURM_JOB_NODELIST"

echo
echo "Python:"
python --version

echo
echo "mpirun:"
which mpirun


# ------------------------------------------------------------
# Confirm rank placement
# ------------------------------------------------------------

echo
echo "============================================================"
echo "MPI Rank Placement"
echo "============================================================"

mpirun --bind-to core \
    --map-by "ppr:${RANKS_PER_NODE}:node" \
    -np "$SLURM_NTASKS" \
    hostname | sort | uniq -c


# ------------------------------------------------------------
# Run final real-data benchmark
# ------------------------------------------------------------

echo
echo "============================================================"
echo "714-Window Real-Data Benchmark"
echo "============================================================"

mpirun \
    --bind-to core \
    --map-by "ppr:${RANKS_PER_NODE}:node" \
    -np "$SLURM_NTASKS" \
    python tests/ninapro_mpi_realdata_benchmark.py \
        --iterations 5 \
        --windows-per-segment 7 \
        --results-file "$RESULTS_FILE"


# ------------------------------------------------------------
# Show final result
# ------------------------------------------------------------

echo
echo "============================================================"
echo "REAL-DATA MPI BENCHMARK COMPLETE"
echo "============================================================"
echo
echo "Results:"
cat "$RESULTS_FILE"