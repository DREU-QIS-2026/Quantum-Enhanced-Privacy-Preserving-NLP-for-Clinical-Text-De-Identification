#!/bin/bash -l

#SBATCH --job-name=ninapro-gpu-final
#SBATCH --account=cis260310p
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --gpus=v100-32:8
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/ninapro-gpu-final-%j.out

set -e

cd ~/QuantumHPC

mkdir -p results/logs
mkdir -p results/plots

# ------------------------------------------------------------
# Prevent hidden CPU threading
# ------------------------------------------------------------

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Use exactly one of the eight allocated V100 GPUs.
export CUDA_VISIBLE_DEVICES=0

# ------------------------------------------------------------
# Load environment
# ------------------------------------------------------------

module load anaconda3/2024.10-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate quantumhpc-gpu

# ------------------------------------------------------------
# Environment information
# ------------------------------------------------------------

echo "============================================================"
echo "NinaPro Final Bridges-2 GPU Benchmark"
echo "============================================================"
echo
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURM_JOB_NODELIST"
echo "Start time   : $(date)"
echo
echo "Python:"
python --version

echo
echo "Qiskit:"
python -c "import qiskit; print(qiskit.__version__)"

echo
echo "Aer:"
python -c "import qiskit_aer; print(qiskit_aer.__version__)"

echo
echo "Visible GPU:"
nvidia-smi -L

echo
echo "Aer devices/methods:"
python -c "from qiskit_aer import AerSimulator; s=AerSimulator(); print('Devices:', s.available_devices()); print('Methods:', s.available_methods())"

# ------------------------------------------------------------
# Remove smoke-test output
# ------------------------------------------------------------

rm -f results/logs/ninapro_bridges_gpu_benchmark.csv
rm -f results/logs/qubit_scaling_bridges_gpu.csv

# ------------------------------------------------------------
# Final benchmark
# ------------------------------------------------------------

echo
echo "============================================================"
echo "Running Final Bridges-2 GPU Experiment"
echo "============================================================"

python tests/ninapro_bridges_gpu_benchmark.py \
    --mode both \
    --backend all \
    --runs 5

# ------------------------------------------------------------
# Show result files
# ------------------------------------------------------------

echo
echo "============================================================"
echo "FINAL GPU BENCHMARK COMPLETE"
echo "============================================================"

echo
echo "NinaPro results:"
cat results/logs/ninapro_bridges_gpu_benchmark.csv

echo
echo "Qubit-scaling results:"
cat results/logs/qubit_scaling_bridges_gpu.csv

echo
echo "End time: $(date)"