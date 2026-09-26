#!/bin/bash

#SBATCH --job-name=ninapro-final
#SBATCH --account=tra260010p
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/ninapro-final-%j.out

set -e

echo "============================================================"
echo "NinaPro Final Bridges-2 Benchmark"
echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node list    : $SLURM_JOB_NODELIST"
echo "Tasks        : $SLURM_NTASKS"
echo "Start time   : $(date)"
echo

# ------------------------------------------------------------
# Thread control
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

echo "Python:"
python --version

echo
echo "mpi4py:"
python -c "import mpi4py; print(mpi4py.__version__)"

echo
echo "mpirun:"
which mpirun

echo
echo "============================================================"
echo "Preparing final result files"
echo "============================================================"

mkdir -p results/logs
mkdir -p results/plots

# The final strong/weak files are new files.
# Remove previous partial final runs so analysis sees only
# this final benchmark.
rm -f results/logs/ninapro_mpi_strong_scaling.csv
rm -f results/logs/ninapro_mpi_weak_scaling.csv

echo
echo "============================================================"
echo "1. NinaPro Noise Validation"
echo "============================================================"

python tests/ninapro_noise_benchmark.py

echo
echo "============================================================"
echo "2. NinaPro Strong Scaling"
echo "============================================================"

for PROCESSES in 1 2 4 8
do

    echo
    echo "------------------------------------------------------------"
    echo "Strong scaling: $PROCESSES MPI processes"
    echo "------------------------------------------------------------"

    mpirun \
        --bind-to core \
        -np "$PROCESSES" \
        python tests/ninapro_mpi_benchmark.py \
        --mode strong \
        --iterations 5 \
        --results-file results/logs/ninapro_mpi_strong_scaling.csv

done

echo
echo "============================================================"
echo "3. NinaPro Weak Scaling"
echo "============================================================"

for PROCESSES in 1 2 4 8
do

    echo
    echo "------------------------------------------------------------"
    echo "Weak scaling: $PROCESSES MPI processes"
    echo "------------------------------------------------------------"

    mpirun \
        --bind-to core \
        -np "$PROCESSES" \
        python tests/ninapro_mpi_benchmark.py \
        --mode weak \
        --iterations 5 \
        --results-file results/logs/ninapro_mpi_weak_scaling.csv

done

echo
echo "============================================================"
echo "4. Generate Final Analysis Plots"
echo "============================================================"

python analysis/analyze_ninapro_noise.py

python analysis/analyze_ninapro_mpi.py

echo
echo "============================================================"
echo "FINAL BENCHMARK COMPLETE"
echo "============================================================"

echo
echo "Result files:"
echo "  results/logs/ninapro_noise_samples.csv"
echo "  results/logs/ninapro_noise_summary.csv"
echo "  results/logs/ninapro_mpi_strong_scaling.csv"
echo "  results/logs/ninapro_mpi_weak_scaling.csv"

echo
echo "Plot files:"
echo "  results/plots/ninapro_noise_fidelity.png"
echo "  results/plots/ninapro_noise_runtime.png"
echo "  results/plots/ninapro_mpi_strong_runtime.png"
echo "  results/plots/ninapro_mpi_strong_speedup.png"
echo "  results/plots/ninapro_mpi_strong_efficiency.png"
echo "  results/plots/ninapro_mpi_weak_runtime.png"
echo "  results/plots/ninapro_mpi_weak_efficiency.png"

echo
echo "End time: $(date)"