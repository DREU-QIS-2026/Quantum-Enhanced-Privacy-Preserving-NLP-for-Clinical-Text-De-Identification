#!/bin/bash

#SBATCH --job-name=quantumhpc-sweep
#SBATCH --account=cis260310p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --array=0-11
#SBATCH --output=results/logs/slurm-%A_%a.out

cd ~/QuantumHPC

# ============================================================
# QuantumHPC Parameter Sweep
#
# 12 total experiments:
#   epsilon = 1, 2, 4, 8
#   conditions = noiseless, depolarizing, amplitude damping
#
# Each SLURM array task represents one experiment.
# ============================================================

# Use the MPI environment verified on Bridges-2.
module load openmpi/4.0.5-nvhpc21.2
module load anaconda3/2024.10-1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate quantumhpc

echo "Python:"
which python
python --version

# Map each array index to an epsilon/condition combination.
EPSILONS=(1 1 1 2 2 2 4 4 4 8 8 8)

CONDITIONS=(
    "noiseless"
    "depolarizing"
    "amplitude_damping"
    "noiseless"
    "depolarizing"
    "amplitude_damping"
    "noiseless"
    "depolarizing"
    "amplitude_damping"
    "noiseless"
    "depolarizing"
    "amplitude_damping"
)

EPSILON=${EPSILONS[$SLURM_ARRAY_TASK_ID]}
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

# ------------------------------------------------------------
# Task-specific result file
# ------------------------------------------------------------

RESULT_FILE="results/logs/slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

export QUANTUMHPC_RESULTS_FILE="$RESULT_FILE"

# ------------------------------------------------------------
# Experiment information
# ------------------------------------------------------------

echo "============================================================"
echo "QuantumHPC SLURM Experiment"
echo "============================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Array Job:    $SLURM_ARRAY_JOB_ID"
echo "Array Task:   $SLURM_ARRAY_TASK_ID"
echo "Node:         $(hostname)"
echo "Epsilon:      $EPSILON"
echo "Condition:    $CONDITION"
echo "========================================"

# ------------------------------------------------------------
# Execute QuantumHPC experiment
# ------------------------------------------------------------

python main.py legacy \
    --epsilon "$EPSILON" \
    --condition "$CONDITION"

# ------------------------------------------------------------
# Completion
# ------------------------------------------------------------

EXIT_CODE=$?

echo "============================================================"
echo "QuantumHPC experiment finished"
echo "Exit code    : $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE