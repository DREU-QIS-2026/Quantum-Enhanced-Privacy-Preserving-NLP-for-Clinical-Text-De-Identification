#!/bin/bash -l

#SBATCH --job-name=ninapro-full-smoke
#SBATCH --account=cis260310p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/logs/ninapro-full-smoke-%j.out

set -e

cd ~/QuantumHPC

mkdir -p results/logs

RESULTS_FILE="results/logs/ninapro_mpi_fullcapacity_smoke.csv"

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
# Environment information
# ------------------------------------------------------------

echo "============================================================"
echo "NinaPro Full-Capacity Real-Data Smoke Test"
echo "============================================================"
echo
echo "Job ID      : $SLURM_JOB_ID"
echo "Node list   : $SLURM_JOB_NODELIST"
echo "MPI ranks   : $SLURM_NTASKS"

echo
echo "Python:"
python --version

echo
echo "mpirun:"
which mpirun


# ------------------------------------------------------------
# One-rank, one-iteration smoke test
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Running Full-Capacity Smoke Test"
echo "============================================================"

mpirun \
    --bind-to core \
    -np 1 \
    python tests/ninapro_mpi_fullcapacity_benchmark.py \
        --iterations 1 \
        --results-file "$RESULTS_FILE"


# ------------------------------------------------------------
# Show final result
# ------------------------------------------------------------

echo
echo "============================================================"
echo "FULL-CAPACITY SMOKE TEST COMPLETE"
echo "============================================================"
echo
echo "Results:"
cat "$RESULTS_FILE"