# QuantumHPC

QuantumHPC is a Distributed Research Experiences for Undergraduates (DREU-QIS) research project investigating scalable quantum simulation, quantum-kernel methods, and high-performance computing (HPC) for biomedical data analysis.

The repository began as part of a collaborative project on quantum-enhanced privacy-preserving NLP for clinical text de-identification. The `kyle-hpc` branch focuses on the HPC and scalability component of that work, including CPU/GPU quantum simulation, MPI parallelization, noise characterization, and real biomedical-data benchmarking.

The final executable HPC study uses NinaPro DB2 surface electromyography (EMG) data as a real biomedical workload.

---

## Research Focus

The HPC work in this branch evaluates several related questions:

- How accurately can manually constructed quantum feature maps be simulated across different backends?
- When does GPU acceleration become useful for small-to-moderate quantum simulations?
- How effectively can pairwise quantum-kernel construction be parallelized with MPI?
- How does physical node placement affect MPI performance?
- How does workload size change the usefulness of multi-node execution?
- How robust is the simulated quantum feature map under controlled depolarizing noise?

The project evaluates these questions using local CPU/GPU hardware and the Pittsburgh Supercomputing Center Bridges-2 system.

---

## Final Experimental Pipeline

The biomedical benchmark uses NinaPro DB2 Subject 1, Exercise 1, Acquisition 1 (`S1_E1_A1.mat`).

Each 512-sample EMG window is processed using:

1. Mean Absolute Value (MAV)
2. Root Mean Square (RMS)
3. Zero Crossings (ZC)
4. StandardScaler normalization
5. PCA reduction from 36 classical features to 8 dimensions
6. MinMax scaling to $[0,\pi]$
7. Manual 8-qubit ZZFeatureMap encoding
8. Quantum-kernel construction using

```math
K(x_i,x_j)
=
\left|\langle \psi(x_i) \mid \psi(x_j)\rangle\right|^2
```

The quantum feature map is implemented manually using primitive Qiskit gates rather than the high-level Qiskit feature-map library.

---

## NinaPro Workloads

Three real-data workload sizes were used during the final MPI study.

| Workload | Real EMG Windows | Kernel Pairs | Purpose |
|---|---:|---:|---|
| Primary | 102 | 5,253 | CPU/GPU validation, noise, initial MPI strong/weak scaling |
| Balanced large workload | 714 | 255,255 | Multi-node workload-size study |
| Full-capacity selected recording | 1,404 | 986,310 | Final process and physical-node scaling |

The 1,404-window workload contains every complete non-overlapping 512-sample window available from the selected `S1_E1_A1.mat` recording.

**Important:** the 1,404 windows represent the maximum complete non-overlapping-window capacity of this selected recording, not the complete NinaPro DB2 dataset across all subjects and exercises.

The raw NinaPro dataset is not included in this repository.

---

## Final Results

### Local CPU/GPU Quantum Simulation

Local benchmarking was performed on:

- Intel Core i7-9700K CPU
- NVIDIA GeForce RTX 2080 SUPER
- qsim CUDA GPU simulation

For the full 102-sample, 8-qubit quantum kernel:

| Backend | Mean Runtime |
|---|---:|
| Local CPU | 0.219500 s |
| Local RTX 2080 SUPER GPU | 0.346955 s |

For this small 8-qubit workload, CPU execution was faster because GPU overhead dominated the computation.

A controlled 4--16-qubit scaling experiment showed a different trend at the largest tested size. At 16 qubits:

| Backend | Mean Runtime |
|---|---:|
| CPU | 1.068073 s |
| RTX 2080 SUPER GPU | 0.438449 s |

This corresponds to an approximately **2.436x GPU speedup** at 16 qubits in the local qsim experiment.

---

### Numerical CPU/GPU Validation

The complete 102-by-102 NinaPro quantum-kernel matrix was computed using both CPU and GPU simulation.

The maximum observed CPU/GPU kernel difference was approximately:

```text
4.77e-07
```

The matrices also passed:

- symmetry checks
- unit-diagonal checks
- valid-range checks
- finite-value checks

---

### Depolarizing-Noise Characterization

The 102-sample NinaPro workload was evaluated under depolarizing-noise strengths:

```text
p = 0.001, 0.005, 0.010, 0.020, 0.050
```

Mean fidelity decreased monotonically from approximately:

```text
0.999961
```

at `p = 0.001` to approximately:

```text
0.998039
```

at `p = 0.050`.

Mean simulation runtime remained approximately constant at about 0.042 seconds per sample.

---

### Bridges-2 MPI Strong Scaling

The initial final strong-scaling experiment used the fixed 102-sample workload.

| MPI Processes | Mean Runtime | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 0.168016 s | 1.000x | 100.00% |
| 2 | 0.084332 s | 1.992x | 99.62% |
| 4 | 0.044457 s | 3.779x | 94.48% |
| 8 | 0.024862 s | 6.758x | 84.47% |

The pairwise quantum-kernel structure therefore provided effective MPI task-level parallelism through eight processes.

Weak-scaling experiments were also completed using approximately constant kernel-pair work per MPI rank.

---

## Full-Capacity MPI Scaling

The final MPI experiment used:

```text
1,404 distinct real EMG windows
986,310 quantum-kernel pairs
8 quantum features / qubits
up to 128 MPI processes
up to 16 physical Bridges-2 nodes
```

### Single-Node Process Scaling

| MPI Ranks | Nodes | Mean Runtime | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 1 | 4.198034 s | 1.000x | 100.00% |
| 8 | 1 | 0.818761 s | 5.127x | 64.09% |
| 16 | 1 | 0.565091 s | 7.429x | 46.43% |
| 32 | 1 | 0.448628 s | 9.357x | 29.24% |
| 64 | 1 | 0.437852 s | 9.588x | 14.98% |
| 128 | 1 | 0.378437 s | 11.093x | 8.67% |

Runtime continued to decrease through 128 MPI processes, although declining parallel efficiency shows substantial diminishing returns at high process counts.

### Fixed-128-Rank Physical-Node Scaling

The total process count was then held constant at 128 while the ranks were distributed across additional physical nodes.

| MPI Ranks | Nodes | Ranks/Node | Mean Runtime | Change vs. 1 Node |
|---:|---:|---:|---:|---:|
| 128 | 1 | 128 | 0.378437 s | baseline |
| 128 | 2 | 64 | 0.389714 s | +2.98% |
| 128 | 4 | 32 | 0.369076 s | -2.47% |
| 128 | 8 | 16 | 0.367011 s | -3.02% |
| 128 | 16 | 8 | 0.365481 s | -3.42% |

The fastest measured full-capacity configuration used:

```text
128 MPI ranks
16 physical Bridges-2 nodes
8 ranks per node
0.365481 s mean runtime
11.486x speedup relative to one MPI process
```

The node-topology differences were modest relative to the overall process-count speedup. The results indicate that larger workloads were able to amortize inter-node communication overhead much more effectively than the smaller workloads.

---

## Bridges-2 GPU Benchmark

A final accelerator experiment was performed on an exclusive Bridges-2 GPU node containing NVIDIA Tesla V100-32GB GPUs.

The benchmark compared:

- Qiskit Aer CPU statevector
- Qiskit Aer V100 GPU statevector
- Qiskit Aer GPU tensor-network simulation using the installed cuQuantum stack

Each configuration used one warm-up followed by five timed runs.

### Full 102-Sample Kernel

| Backend | Mean Runtime | CPU-Relative Speedup |
|---|---:|---:|
| Aer CPU statevector | 7.132028 s | 1.000x |
| V100 GPU statevector | 7.210690 s | 0.989x |
| V100 GPU tensor network | 8.277748 s | 0.862x |

The complete 8-qubit workload remained too small for the V100 to outperform CPU execution.

### Qubit Scaling

The same Bridges-2 environment was evaluated from 4 through 16 qubits.

CPU and V100 statevector performance remained close through 14 qubits. At 16 qubits:

```text
Aer CPU statevector:   0.634693 s
V100 GPU statevector:  0.594701 s
GPU speedup:           1.067x
```

The V100 therefore became faster than the CPU at the largest tested circuit size.

The tensor-network backend did not outperform CPU statevector simulation within the tested 4--16-qubit range.

These results are specific to the tested circuits and should not be interpreted as a universal CPU/GPU or statevector/tensor-network crossover threshold.

---

## Repository Structure

```text
QuantumHPC/
|
|-- analysis/        Result-analysis and plotting scripts
|-- backends/        Backend configuration utilities
|-- benchmark/       General benchmarking utilities
|-- circuits/        Manual quantum feature-map implementations
|-- data/            Local dataset location; raw NinaPro data is not tracked
|-- datasets/        NinaPro loading and preprocessing
|-- docs/            Literature review, manuscript, and research documentation
|-- logs/            Weekly DREU research logs
|-- mpi/             MPI-related utilities and prototypes
|-- results/
|   |-- logs/        Benchmark CSVs and selected job output
|   `-- plots/       Generated experiment figures
|-- simulation/      Quantum-kernel and simulator implementations
|-- slurm/           Bridges-2 SLURM batch scripts
|-- tests/           Experimental benchmark drivers and validation scripts
|
|-- main.py
|-- config.py
|-- requirements.txt
|-- requirements-hpc.txt
|-- README.md
|-- ROADMAP.md
|-- CHANGELOG.md
|-- TEAM_SETUP.md
`-- CONTRIBUTING.md
```

The `tests/` directory contains several research benchmark drivers in addition to conventional validation scripts. Final experiment execution is performed through these benchmark drivers and the corresponding SLURM workflows rather than through `main.py`.

---

## Important Experiment Files

### NinaPro and Quantum Simulation

```text
datasets/ninapro.py
simulation/quantum_kernel.py
simulation/qsim_simulator.py
circuits/feature_maps.py
```

### Final Benchmark Drivers

```text
tests/ninapro_benchmark.py
tests/ninapro_noise_benchmark.py
tests/ninapro_mpi_benchmark.py
tests/ninapro_mpi_realdata_benchmark.py
tests/ninapro_mpi_fullcapacity_benchmark.py
tests/ninapro_bridges_gpu_benchmark.py
```

### Final Analysis Scripts

```text
analysis/analyze_ninapro_noise.py
analysis/analyze_ninapro_mpi.py
analysis/analyze_ninapro_bridges_gpu.py
analysis/analyze_ninapro_fullcapacity_mpi.py
```

### Bridges-2 Workflows

Representative SLURM scripts are stored in:

```text
slurm/
```

These include the final CPU MPI, multi-node, full-capacity, smoke-test, and GPU benchmarking workflows.

---

## Environment

### Core Software

The project uses:

- Python
- NumPy
- SciPy
- pandas
- scikit-learn
- matplotlib
- Qiskit
- Qiskit Aer
- mpi4py
- OpenMPI
- Cirq/qsim for the local CUDA path
- NVIDIA CUDA
- NVIDIA cuQuantum for the Bridges-2 tensor-network experiment
- SLURM on Bridges-2

The final Bridges-2 CPU/MPI environment used:

```text
Python 3.12.13
Qiskit 2.5.0
Qiskit Aer 0.17.2
mpi4py 4.1.2
OpenMPI 4.0.5
```

The Bridges-2 GPU environment used Qiskit Aer GPU support with CUDA 11 dependencies and the cuQuantum software stack.

---

## Installation

Clone the repository and switch to the HPC branch.

```bash
git clone https://github.com/DREU-QIS-2026/Quantum-Enhanced-Privacy-Preserving-NLP-for-Clinical-Text-De-Identification.git
cd Quantum-Enhanced-Privacy-Preserving-NLP-for-Clinical-Text-De-Identification
git checkout kyle-hpc
```

Create a Python environment appropriate for the target system.

Example:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the core dependencies:

```bash
pip install -r requirements.txt
```

HPC-specific Python dependencies are documented in:

```text
requirements-hpc.txt
```

MPI execution also requires a working system MPI implementation such as OpenMPI.

GPU environments require backend-specific CUDA dependencies and should be configured separately from the CPU/MPI environment.

---

## Bridges-2 CPU/MPI Environment

The final Bridges-2 CPU experiments used the following environment setup:

```bash
module load openmpi/4.0.5-nvhpc21.2
module load anaconda3/2024.10-1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate quantumhpc
```

SLURM scripts should be reviewed before reuse because account names, partitions, node limits, and software modules are system- and allocation-specific.

---

## Running Analysis

Once the corresponding result CSVs are available, the final analysis scripts can be run from the repository root.

Noise analysis:

```bash
python analysis/analyze_ninapro_noise.py
```

Original MPI strong/weak-scaling analysis:

```bash
python analysis/analyze_ninapro_mpi.py
```

Bridges-2 accelerator analysis:

```bash
python analysis/analyze_ninapro_bridges_gpu.py
```

Final full-capacity MPI analysis:

```bash
python analysis/analyze_ninapro_fullcapacity_mpi.py
```

Generated figures are written to:

```text
results/plots/
```

and processed CSV summaries are written to:

```text
results/logs/
```

---

## Reproducibility Notes

The repository preserves:

- benchmark source code
- MPI process-distribution logic
- SLURM job scripts
- result CSV files
- plotting and analysis scripts
- selected benchmark logs
- weekly research logs
- manuscript and literature-review materials

The raw NinaPro dataset is intentionally excluded.

To reproduce the biomedical experiments, obtain the NinaPro DB2 dataset separately and place the required file at the path expected by `datasets/ninapro.py`, for example:

```text
data/ninapro/DB2/s1/S1_E1_A1.mat
```

The exact performance values reported here are hardware- and software-dependent. Reproducing the code path should not be interpreted as guaranteeing identical runtime measurements on different systems.

---

## Weekly Research Logs

Weekly and extended project updates are stored in:

```text
logs/
```

The logs document:

- project goals
- implementation decisions
- troubleshooting
- benchmark development
- CPU/GPU validation
- MPI and multi-node experiments
- Bridges-2 deployment
- final project results
- remaining submission tasks

The final project period is documented in the Week 10+ log.

---

## Team

| Member | Primary Project Responsibilities |
|---|---|
| Kyle Hartness | HPC benchmarking, MPI parallelization, CPU/GPU simulation, scalability, NinaPro experiments, repository maintenance |
| Edmund Bombardieri | Quantum circuits, quantum algorithms, QPE/QFI, quantum methods |
| Sofia Furda | Clinical data, preprocessing, NLP pipeline, privacy/utility workflow |

The final HPC benchmarking and NinaPro work documented in this branch represents Kyle Hartness's primary DREU contribution.

---

## Documentation

Additional project documentation includes:

```text
TEAM_SETUP.md
CONTRIBUTING.md
CHANGELOG.md
ROADMAP.md
logs/
docs/
```

The final research manuscript and literature-review materials are maintained under `docs/`.

---

## Project Status

Experimental data collection is complete.

Completed work includes:

- manual quantum feature-map construction
- CPU quantum simulation
- local CUDA GPU simulation
- CPU/GPU numerical validation
- local qubit scaling
- depolarizing-noise characterization
- NinaPro biomedical-data integration
- MPI strong scaling
- MPI weak scaling
- multi-node MPI topology testing
- balanced 714-window real-data scaling
- full-capacity 1,404-window MPI scaling
- execution through 128 MPI ranks
- execution across as many as 16 Bridges-2 nodes
- Bridges-2 V100 statevector benchmarking
- Bridges-2 tensor-network/cuQuantum benchmarking
- final performance analysis and publication figures

Remaining work is limited to final documentation, repository cleanup, manuscript/presentation preparation, and archival synchronization.

---

## Acknowledgements

This work was completed through the Distributed Research Experiences for Undergraduates (DREU) program as part of the DREU-QIS 2026 research project.

HPC experiments were performed using Bridges-2 resources at the Pittsburgh Supercomputing Center through the ACCESS program.

Additional project-specific funding and acknowledgment language is provided in the final manuscript.