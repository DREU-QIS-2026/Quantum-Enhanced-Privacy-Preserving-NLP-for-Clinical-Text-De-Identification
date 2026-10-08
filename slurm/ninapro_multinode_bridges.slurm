#!/bin/bash -l

#SBATCH --job-name=ninapro-multinode
#SBATCH --partition=RM
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/ninapro-multinode-%j.out

set -e

cd ~/QuantumHPC

mkdir -p results/logs


# ------------------------------------------------------------
# Configuration supplied at submission time
# ------------------------------------------------------------

if [ -z "$RESULT_TAG" ]; then
    echo "ERROR: RESULT_TAG was not supplied."
    exit 1
fi

RESULTS_FILE="results/logs/ninapro_mpi_multinode_${RESULT_TAG}.csv"

rm -f "$RESULTS_FILE"


# ------------------------------------------------------------
# Prevent hidden threading inside each MPI rank
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
# Experiment information
# ------------------------------------------------------------

RANKS_PER_NODE=$(( SLURM_NTASKS / SLURM_JOB_NUM_NODES ))

echo "============================================================"
echo "NinaPro Multi-Node MPI Extension"
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
# Confirm MPI rank placement
# ------------------------------------------------------------

echo
echo "============================================================"
echo "MPI Rank Placement"
echo "============================================================"

mpirun --bind-to core --map-by "ppr:${RANKS_PER_NODE}:node" -np "$SLURM_NTASKS" hostname | sort | uniq -c

# ------------------------------------------------------------
# Run fixed-workload strong-scaling benchmark
# ------------------------------------------------------------

echo
echo "============================================================"
echo "NinaPro Strong-Scaling Benchmark"
echo "============================================================"

mpirun \
    --bind-to core \
    --map-by ppr:${RANKS_PER_NODE}:node \
    -np "$SLURM_NTASKS" \
    python tests/ninapro_mpi_benchmark.py \
        --mode strong \
        --iterations 5 \
        --results-file "$RESULTS_FILE"


# ------------------------------------------------------------
# Show result
# ------------------------------------------------------------

echo
echo "============================================================"
echo "MULTI-NODE BENCHMARK COMPLETE"
echo "============================================================"

echo
echo "Results:"
cat "$RESULTS_FILE"