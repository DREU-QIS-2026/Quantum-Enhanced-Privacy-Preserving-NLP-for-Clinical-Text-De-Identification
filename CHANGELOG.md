# Changelog

This changelog records the major development milestones of the `kyle-hpc` branch of the QuantumHPC DREU-QIS 2026 project.

The branch evolved from an initial general-purpose quantum/HPC framework into a completed experimental workflow for biomedical quantum-kernel simulation using NinaPro EMG data, CPU/GPU acceleration, MPI parallelism, multi-node Bridges-2 execution, noise characterization, and final publication analysis.

---

## Unreleased — Final DREU Project State

This section represents the completed local project state prior to final repository cleanup and synchronization.

### Added

- Full-capacity NinaPro MPI benchmark using every complete non-overlapping 512-sample window available from the selected `S1_E1_A1.mat` recording
- Final 1,404-window real-data workload
- Final 986,310-pair quantum-kernel workload
- MPI process scaling through 128 ranks
- Physical-node scaling through 16 Bridges-2 Regular Memory nodes
- Fixed-128-rank node-topology experiment
- Final full-capacity MPI analysis script
- Final full-capacity MPI summary CSV
- Final fixed-rank topology summary CSV
- Publication-quality full-capacity process-scaling figure
- Publication-quality fixed-128-rank physical-node-scaling figure
- Final Bridges-2 V100 accelerator analysis
- Final Bridges-2 accelerator summary CSVs
- Final Bridges-2 GPU runtime and speedup figures
- Week 10+ project wrap-up research log

### Final Full-Capacity MPI Results

The final MPI workload contains:

```text
1,404 distinct real NinaPro EMG windows
986,310 upper-triangular quantum-kernel pairs
8 quantum features / qubits
```

Single-node process scaling:

| MPI Ranks | Nodes | Mean Runtime | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 1 | 4.198034 s | 1.000x | 100.00% |
| 8 | 1 | 0.818761 s | 5.127x | 64.09% |
| 16 | 1 | 0.565091 s | 7.429x | 46.43% |
| 32 | 1 | 0.448628 s | 9.357x | 29.24% |
| 64 | 1 | 0.437852 s | 9.588x | 14.98% |
| 128 | 1 | 0.378437 s | 11.093x | 8.67% |

Fixed-128-rank physical-node scaling:

| MPI Ranks | Nodes | Ranks/Node | Mean Runtime | Change vs. 1 Node |
|---:|---:|---:|---:|---:|
| 128 | 1 | 128 | 0.378437 s | baseline |
| 128 | 2 | 64 | 0.389714 s | +2.98% |
| 128 | 4 | 32 | 0.369076 s | -2.47% |
| 128 | 8 | 16 | 0.367011 s | -3.02% |
| 128 | 16 | 8 | 0.365481 s | -3.42% |

The fastest measured configuration used 128 MPI ranks distributed across 16 physical nodes and achieved approximately `11.486x` speedup relative to the single-process baseline.

### Changed

- Expanded the final manuscript to make sample count, MPI process count, and physical-node count explicit
- Promoted the 1,404-window workload to the primary final multi-node MPI result
- Retained the earlier 102-window and 714-window workloads to show how workload size changes communication behavior
- Updated README to reflect the completed experimental project rather than the early development plan
- Updated ROADMAP to reflect completed CPU, GPU, MPI, Bridges-2, NinaPro, and analysis milestones

### Fixed

- Corrected final documentation that previously described the 714-window experiment as the largest real-data workload
- Corrected documentation that previously described four physical nodes and 32 MPI ranks as the largest distributed configuration
- Corrected project documentation that still described MPI and Bridges-2 work as incomplete

---

## Phase 10 — Bridges-2 GPU Final Benchmark

### Added

- Dedicated Bridges-2 GPU benchmark driver
- Separate GPU Conda environment to avoid modifying the established CPU/MPI environment
- Qiskit Aer GPU support
- CUDA 11 runtime dependencies
- cuQuantum dependencies
- Aer GPU statevector validation
- Aer GPU tensor-network validation
- Exclusive-node final Bridges-2 GPU benchmark
- Controlled 4--16-qubit Bridges-2 accelerator scaling
- Repeated timing measurements using one warm-up and five timed runs
- Final GPU analysis and plotting workflow

### Final 102-Sample Bridges-2 Results

| Backend | Mean Runtime | CPU-Relative Speedup |
|---|---:|---:|
| Aer CPU statevector | 7.132028 s | 1.000x |
| V100 GPU statevector | 7.210690 s | 0.989x |
| V100 GPU tensor network | 8.277748 s | 0.862x |

For the complete 102-sample, 8-qubit kernel, GPU acceleration did not improve runtime.

### Final Qubit-Scaling Result

At 16 qubits:

```text
Aer CPU statevector:  0.634693 s
V100 GPU statevector: 0.594701 s
V100 speedup:          1.067x
```

The V100 statevector backend became faster than CPU execution at the largest tested circuit size.

The tensor-network backend did not outperform CPU statevector simulation over the tested 4--16-qubit range.

### Fixed

- Resolved lack of GPU support in the original Bridges-2 CPU environment by creating a dedicated GPU environment
- Installed the CUDA-11-compatible Qiskit Aer GPU package and cuQuantum dependency stack
- Verified both `statevector` and `tensor_network` GPU methods before final benchmarking
- Avoided direct modification of the established CPU/MPI environment

---

## Phase 9 — Full-Capacity Real-Data MPI Extension

### Added

- `tests/ninapro_mpi_fullcapacity_benchmark.py`
- Full-capacity smoke-test SLURM workflow
- Full-capacity final Bridges-2 SLURM workflow
- Automatic validation of:
  - segment count
  - window count
  - feature count
  - qubit count
  - kernel-pair count
  - output kernel matrix
- Support for variable numbers of real windows per gesture/repetition segment
- Full-capacity MPI configurations from 1 through 128 ranks
- Fixed-128-rank topology configurations from 1 through 16 physical nodes

### Workload

The selected NinaPro recording contains:

```text
102 gesture/repetition segments
7–27 complete windows per segment
13.76 mean windows per segment
1,404 total distinct real windows
986,310 quantum-kernel pairs
```

### Improved

- Increased the largest MPI workload from 714 to 1,404 real windows
- Increased kernel-pair work from 255,255 to 986,310 evaluations
- Increased maximum MPI rank count from 32 to 128
- Increased maximum physical-node count from 4 to 16
- Separated process-count scaling from physical-node scaling

---

## Phase 8 — Larger Real-Data Multi-Node MPI

### Added

- Analysis of the available real-window capacity in the selected NinaPro recording
- Balanced 714-window real-data workload
- Seven distinct non-overlapping windows per gesture/repetition segment
- Multi-node MPI experiments using the larger workload
- Matched single-node and multi-node process-count comparisons

### Workload

```text
17 gestures
6 repetitions
7 windows per gesture/repetition
714 real EMG windows
255,255 quantum-kernel pairs
```

### Results

The 714-window workload demonstrated that increasing computational work substantially reduced the relative effect of inter-node communication.

Matched single-node and multi-node runtimes differed by less than approximately 0.4% in the tested 8-, 16-, and 32-process configurations.

### Improved

- Replaced deterministic feature reuse as the primary large-workload topology analysis with distinct real EMG windows
- Established a clearer relationship between workload size and multi-node MPI behavior

---

## Phase 7 — Multi-Node Bridges-2 MPI Extension

### Added

- Dedicated multi-node Bridges-2 SLURM workflow
- Explicit MPI process placement with `--map-by`
- Rank-placement verification using node hostnames
- One-node vs. multi-node comparisons at matched MPI process counts

### Tested Configurations

```text
8 ranks / 1 node
8 ranks / 2 nodes

16 ranks / 1 node
16 ranks / 2 nodes

32 ranks / 1 node
32 ranks / 4 nodes
```

### Results

The original 102-sample workload showed measurable inter-node communication overhead when the same MPI process count was spread across additional physical nodes.

This result motivated the larger 714-window and final 1,404-window experiments.

### Fixed

- Corrected early assumptions that MPI process scaling alone demonstrated multi-node execution
- Added explicit node-placement verification to ensure the intended topology was actually used
- Resolved SLURM script syntax and line-ending issues during multi-node workflow development

---

## Phase 6 — Final Bridges-2 Strong and Weak Scaling

### Added

- Final NinaPro MPI benchmark driver
- Strong-scaling mode
- Weak-scaling mode
- MPI state encoding
- MPI state exchange
- Distributed upper-triangular kernel-pair computation
- Kernel result gathering
- Full symmetric kernel reconstruction
- Runtime, speedup, and efficiency analysis
- Strong-scaling plots
- Weak-scaling plots
- SLURM final benchmark workflow

### Strong-Scaling Results

Using the 102-sample workload:

| MPI Processes | Mean Runtime | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 0.168016 s | 1.000x | 100.00% |
| 2 | 0.084332 s | 1.992x | 99.62% |
| 4 | 0.044457 s | 3.779x | 94.48% |
| 8 | 0.024862 s | 6.758x | 84.47% |

### Weak Scaling

Weak scaling maintained approximately constant quantum-kernel pair work per MPI rank.

The workload size increased according to the all-pairs kernel relationship rather than by linearly increasing the number of input samples.

### Fixed

- Resolved batch-job failure caused by the `module` command not being available in the original non-login SLURM shell
- Updated SLURM execution to use a login shell
- Verified Bridges-2 Python, Qiskit, Aer, OpenMPI, and `mpi4py` environments
- Separated benchmark timing from environment setup and result analysis

---

## Phase 5 — Bridges-2 Access Recovery

### Changed

- Previous temporary workshop allocation became unavailable for final SLURM submission
- Documented final HPC resource requirements for the faculty mentor
- Requested Bridges-2 Regular Memory CPU resources for MPI work
- Requested Bridges-2 GPU resources for accelerator comparison
- Restored project execution through a new ACCESS allocation

### Added

- Final ACCESS resource summary
- Regular Memory resource requirements
- GPU resource requirements
- Two-week final project execution plan

### Result

Final Bridges-2 CPU/MPI and GPU experiments were completed after access was restored.

---

## Phase 4 — NinaPro Biomedical Benchmark

### Added

- NinaPro DB2 dataset support
- Subject 1, Exercise 1, Acquisition 1 workflow
- 512-sample EMG window extraction
- Mean Absolute Value feature extraction
- Root Mean Square feature extraction
- Zero Crossing feature extraction
- 36-feature classical EMG representation
- StandardScaler normalization
- PCA reduction to 8 dimensions
- MinMax scaling to \([0,\pi]\)
- 8-dimensional quantum input representation
- Original 102-window gesture/repetition workload
- Complete 102-by-102 quantum-kernel generation

### Primary Workload

```text
17 gestures
6 repetitions
1 window per gesture/repetition
102 real samples
5,253 upper-triangular kernel pairs
8 quantum features / qubits
```

### Improved

- Replaced synthetic/sample-only performance testing with a real biomedical workload
- Standardized all final CPU, GPU, noise, and MPI experiments around the same quantum-feature construction pipeline

---

## Phase 3 — Local CPU/GPU Quantum Simulation

### Added

- Local CPU quantum-kernel benchmark
- Local qsim simulation backend
- CUDA-enabled qsim build
- RTX 2080 SUPER GPU execution
- CPU/GPU numerical kernel validation
- Qubit-scaling benchmark
- Runtime plots
- CPU/GPU speedup analysis

### Numerical Validation

The complete CPU and GPU 102-by-102 kernel matrices agreed with maximum absolute difference of approximately:

```text
4.77e-07
```

All matrices passed:

- symmetry validation
- diagonal validation
- range validation
- finite-value validation

### Full 102-Sample Runtime

```text
CPU mean runtime: 0.219500 s
GPU mean runtime: 0.346955 s
```

For the tested 8-qubit workload, CPU execution was faster.

### Qubit Scaling

The controlled qubit-scaling benchmark evaluated:

```text
4, 6, 8, 10, 12, 14, 16 qubits
```

At 16 qubits:

```text
CPU mean runtime: 1.068073 s
GPU mean runtime: 0.438449 s
GPU speedup:      2.436x
```

### Fixed

- Corrected GPU provider and backend selection
- Validated numerical agreement between Qiskit and qsim-based implementations
- Avoided use of WSL by using the native Windows qsim CUDA path

---

## Phase 2 — Noise Characterization

### Added

- Depolarizing-noise experiment
- Density-matrix simulation workflow
- Noise-strength sweep
- Fidelity analysis
- Per-sample results CSV
- Summary results CSV
- Fidelity plot
- Runtime plot

### Tested Noise Strengths

```text
0.001
0.005
0.010
0.020
0.050
```

### Results

Mean fidelity decreased monotonically from approximately:

```text
0.999961
```

to:

```text
0.998039
```

over the tested noise range.

Mean runtime remained approximately:

```text
0.042 s/sample
```

across the tested configurations.

---

## Phase 1 — Project Framework

### Added

- Modular project structure
- DatasetLoader supporting CSV, TSV, Excel, JSON, and Parquet
- DataPreprocessor for numeric feature extraction
- FeatureMapBuilder using Qiskit's ZZFeatureMap
- Manual ZZFeatureMap implementation using primitive quantum gates
- ExperimentLogger with CSV output
- Runtime benchmarking
- Automatic circuit PNG generation
- Clinical sample dataset
- Clinical notes sample dataset
- Initial Qiskit Aer simulation framework
- Initial statevector simulation
- GitHub collaboration workflow
- Weekly DREU research logs

### Improved

- Automatic numeric feature detection
- Dynamic qubit count based on dataset features
- Organized plot and result directories
- Manual vs. library circuit comparison
- Reproducible experiment organization

---

## Final Project Outcome

The `kyle-hpc` branch now contains a completed HPC-focused biomedical quantum-simulation study with:

- manual quantum feature-map construction
- real NinaPro EMG data
- CPU statevector simulation
- local CUDA GPU simulation
- CPU/GPU numerical validation
- controlled qubit scaling
- depolarizing-noise characterization
- MPI strong scaling
- MPI weak scaling
- multi-node MPI execution
- balanced larger real-data testing
- full-capacity real-data MPI testing
- execution through 128 MPI ranks
- execution across as many as 16 Bridges-2 nodes
- Bridges-2 NVIDIA V100 acceleration
- Qiskit Aer GPU statevector simulation
- cuQuantum-backed tensor-network simulation
- reproducible CSV result generation
- publication-quality analysis figures
- research manuscript integration
- extended DREU research logs

Experimental data collection is complete.

Remaining work consists of final repository cleanup, dependency review, reproducibility verification, manuscript/presentation completion, and synchronization of the final branch to both GitHub repositories.