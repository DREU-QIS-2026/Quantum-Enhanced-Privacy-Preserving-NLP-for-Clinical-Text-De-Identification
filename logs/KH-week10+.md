# Week 10+ — Project Wrap-Up

**Date:** September 14 – September 29, 2026

---

# Goals

- Recover Bridges-2 access after the previous workshop allocation became unavailable.
- Complete the final NinaPro MPI strong- and weak-scaling experiments on Bridges-2.
- Extend the MPI evaluation from single-node execution to true multi-node execution.
- Evaluate whether a larger real-data workload can better utilize multiple Bridges-2 nodes.
- Establish and validate a Bridges-2 NVIDIA V100 GPU simulation environment.
- Prepare the final research paper, figures, literature review, and reproducibility materials for project submission.
- Respond to final mentor feedback by making dataset size and physical-node usage explicit and extending the MPI evaluation to the maximum complete non-overlapping-window workload available from the selected NinaPro recording.

---

# Approach & Implementation

The final HPC benchmarking phase initially encountered an access roadblock when the previous Bridges-2 workshop allocation could no longer be used for SLURM job submission. Although the existing repository, software environment, NinaPro dataset, MPI benchmark code, and final SLURM workflow were still present on Bridges-2, attempts to submit the final benchmark failed because the account was no longer associated with an active allocation.

Worked with the faculty mentor to document the required ACCESS resources for the remainder of the project. The request included Bridges-2 Regular Memory resources for the MPI scaling experiments and Bridges-2 GPU resources for an additional accelerator comparison. A new project allocation was subsequently established, restoring access to both Regular Memory and GPU resources.

After access was restored, the final NinaPro MPI workflow was executed on Bridges-2. The benchmark used 102 real NinaPro gesture/repetition samples, eight quantum features, an eight-qubit manual ZZFeatureMap, and 5253 upper-triangular quantum-kernel pairs. Strong-scaling experiments were completed using 1, 2, 4, and 8 MPI processes. Weak-scaling experiments were also completed using approximately constant kernel-pair work per process.

During the first batch submission, the SLURM script failed because the `module` command was unavailable in the non-login batch shell. The batch environment was tested separately, and the final workflow was corrected to use a login shell. After this correction, the complete benchmark executed successfully and generated the final strong- and weak-scaling CSV files and analysis plots.

The MPI evaluation was then extended beyond the original single-node experiment. A dedicated topology benchmark compared matched MPI process counts using one or multiple Bridges-2 Regular Memory nodes. Tests were completed using 8 processes on one and two nodes, 16 processes on one and two nodes, and 32 processes on one and four nodes. MPI rank placement was explicitly controlled and verified from the batch logs to ensure that the intended process distribution was actually used.

The 102-sample topology experiment showed that distributing a small fixed workload across multiple nodes could introduce measurable communication overhead. To determine whether this result was primarily caused by the small workload size, the NinaPro processing workflow was examined for additional real EMG data. The original 102 gesture/repetition segments were found to contain between 7 and 27 non-overlapping 512-sample windows, providing 1404 possible real windows in total.

A balanced larger workload was therefore constructed using exactly seven distinct non-overlapping windows from every gesture/repetition segment. This produced 714 real EMG windows while preserving equal representation across all 17 gestures and six repetitions. The same MAV, RMS, and zero-crossing feature extraction, StandardScaler normalization, PCA reduction to eight dimensions, and MinMax scaling to the interval [0, pi] were applied to the expanded dataset. The resulting 714-sample quantum kernel contained 255255 upper-triangular pair evaluations, approximately 48.6 times more pair evaluations than the original 102-sample workload.

The larger real-data workload was evaluated using 1 process as a baseline and matched single-node and multi-node configurations at 8, 16, and 32 MPI processes. Each configuration used one untimed warm-up and five timed iterations. All kernel matrices passed symmetry, diagonal, range, and finite-value validation.

In parallel with the MPI work, the Bridges-2 GPU environment was configured. A separate Conda environment was created so that the existing CPU/MPI environment would remain unchanged. Qiskit Aer GPU support and the CUDA 11 cuQuantum dependencies were installed and tested on a Bridges-2 NVIDIA Tesla V100-32GB GPU. Both Aer GPU statevector simulation and the GPU `tensor_network` method were successfully validated.

Preliminary V100 smoke tests were completed for the 102-sample NinaPro kernel and for controlled qubit scaling from 4 to 16 qubits. CPU statevector, V100 statevector, and V100 tensor-network execution all produced valid quantum-kernel results. A final repeated benchmark using an exclusive Bridges-2 GPU node was prepared and submitted so that the three backend configurations can be compared under controlled benchmarking conditions.

The final manuscript was also substantially expanded during this period. The literature review and source notes were reviewed and updated, additional quantum-kernel and MPI references were incorporated, and the final paper was updated with the NinaPro methodology, local CPU/GPU results, depolarizing-noise experiment, strong- and weak-scaling MPI results, multi-node MPI methodology, and final real-data scaling experiment. Figure placement and LaTeX formatting were also revised to improve the readability of the final document. The required NSF DREU-QIS acknowledgment language and mentor/author information were identified for inclusion in the final manuscript.

Following review of the near-final manuscript, additional mentor feedback requested that the number of dataset samples and physical compute nodes be made more explicit and that, if feasible, the MPI evaluation use the largest available real-data workload and additional ACCESS nodes. The NinaPro S1_E1_A1 recording was therefore re-evaluated at its maximum complete non-overlapping-window capacity.

Every complete non-overlapping 512-sample window available from the longest contiguous segment for each of the 102 gesture/repetition combinations was retained. Segment lengths supported between 7 and 27 windows, with an average of approximately 13.76 windows per segment. This produced 1,404 distinct real EMG windows and 986,310 upper-triangular quantum-kernel pair evaluations. The workload represents the maximum complete non-overlapping-window capacity of the selected Subject 1, Exercise 1, Acquisition 1 recording rather than the complete NinaPro DB2 dataset across all subjects.

A one-process smoke test was first performed to verify the workload and kernel validation. The smoke test successfully produced all 1,404 expected windows and 986,310 kernel pairs, and the resulting kernel matrix passed validation.

The final full-capacity MPI experiment then evaluated process-count scaling on one Bridges-2 Regular Memory node using 1, 8, 16, 32, 64, and 128 MPI processes. A second topology experiment held the total MPI process count fixed at 128 while distributing those ranks across 1, 2, 4, 8, and 16 physical Bridges-2 nodes. Each configuration used one untimed warm-up followed by five timed iterations. All ten final batch jobs completed successfully with zero exit codes and valid quantum-kernel results.

---

# Results

- Restored Bridges-2 access through a new project allocation with both Regular Memory and GPU resources.
- Completed the final 102-sample NinaPro MPI strong-scaling experiment using 1, 2, 4, and 8 processes.
- Reduced strong-scaling mean runtime from approximately 0.168016 s with one process to approximately 0.024862 s with eight processes.
- Achieved approximately 6.76x strong-scaling speedup with eight MPI processes and approximately 84.47% parallel efficiency.
- Completed the final weak-scaling experiment using 1, 2, 4, and 8 MPI processes.
- Extended MPI testing to true multi-node Bridges-2 execution and explicitly verified process placement across compute nodes.
- Demonstrated that the smaller 102-sample workload experienced increasing inter-node overhead at higher matched process counts.
- Identified 1404 possible non-overlapping real NinaPro windows across the 102 gesture/repetition segments.
- Constructed a balanced larger workload containing 714 distinct real EMG windows, with seven windows selected from every gesture/repetition segment.
- Increased the real-data quantum-kernel workload from 5253 to 255255 upper-triangular kernel pairs.
- Completed seven final configurations for the 714-window workload: 1 process/1 node, 8 processes/1 node, 8 processes/2 nodes, 16 processes/1 node, 16 processes/2 nodes, 32 processes/1 node, and 32 processes/4 nodes.
- Reduced runtime for the 714-window workload from approximately 1.623986 s with one process to approximately 0.136360 s with 32 processes across four nodes.
- Achieved approximately 11.91x speedup with 32 MPI processes relative to the single-process baseline.
- Observed less than 0.4% difference between matched single-node and multi-node runtimes at 8, 16, and 32 MPI processes for the larger workload, indicating that inter-node communication overhead was largely amortized by the increased computation.
- Successfully configured Qiskit Aer GPU support on a Bridges-2 NVIDIA Tesla V100-32GB GPU.
- Successfully validated both Aer GPU statevector and GPU tensor-network/cuQuantum simulation.
- Successfully completed preliminary NinaPro and 4–16-qubit V100 GPU smoke tests with valid quantum-kernel outputs.
- Completed the final exclusive-node Bridges-2 GPU benchmark.
- Evaluated Aer CPU statevector, V100 GPU statevector, and V100 GPU tensor-network/cuQuantum simulation using one warm-up and five timed runs per configuration.
- Confirmed valid quantum-kernel output for every final accelerator benchmark run.
- Measured mean full-kernel runtimes of approximately 7.132 s for Aer CPU statevector, 7.211 s for V100 statevector, and 8.278 s for V100 tensor-network simulation.
- Completed final 4–16-qubit Bridges-2 CPU/GPU scaling measurements.
- Observed near-parity between CPU and V100 statevector execution through 14 qubits and a measured V100 statevector speedup of approximately 1.067x at 16 qubits.
- Confirmed that the tested GPU tensor-network configuration did not outperform CPU statevector simulation over the evaluated 4–16-qubit range.
- Completed all planned experimental data collection for the project.
- Expanded the final LaTeX manuscript with updated methodology, MPI results, multi-node analysis, figures, literature citations, limitations, and discussion material.
- Identified the required NSF DREU-QIS acknowledgment language for the final paper.
- Completed the final exclusive-node Bridges-2 GPU benchmark using Aer CPU statevector, V100 GPU statevector, and V100 GPU tensor-network/cuQuantum simulation.
- Confirmed valid quantum-kernel output for every final Bridges-2 accelerator timing run.
- Measured approximately 7.132 s for the complete 102-sample Aer CPU statevector kernel, 7.211 s for V100 statevector simulation, and 8.278 s for V100 tensor-network simulation.
- Completed final Bridges-2 qubit-scaling measurements from 4 through 16 qubits and observed a measured V100 statevector speedup of approximately 1.067x relative to Aer CPU simulation at 16 qubits.
- Constructed the maximum complete non-overlapping-window workload available from the selected NinaPro S1_E1_A1 recording.
- Confirmed that the full-capacity workload contains 1,404 distinct real EMG windows and 986,310 upper-triangular quantum-kernel pairs.
- Completed final single-node process scaling using 1, 8, 16, 32, 64, and 128 MPI ranks.
- Reduced full-capacity mean runtime from approximately 4.198034 s with one MPI process to 0.378437 s with 128 processes on one node corresponding to approximately 11.093x speedup.
- Completed fixed-128-rank physical-node scaling using 1, 2, 4, 8, and 16 Bridges-2 Regular Memory nodes.
- Measured the fastest full-capacity configuration using 128 MPI ranks distributed across 16 physical nodes at approximately 0.365481s.
- Achieved approximately 11.486x overall speedup for the 128-rank, 16-node configuration relative to the single-process baseline.
- Observed substantial diminishing parallel efficiency at high MPI process counts, with approximately 8.97% efficiency for the 128-rank, 16-node configuration.
- Demonstrated that the effect of physical-node distribution depends strongly on workload size: the smallest workload showed measurable inter-node overhead, the 714-window workload was approximately topology-neutral, and the 1,404-window workload produced comparable or modestly faster runtimes across several multi-node configurations.
- Completed all planned and final mentor-requested experimental data collection.

---

# Next Steps

- Integrate the final full-capacity MPI results and publication figures into the manuscript.
- Complete the remaining literature-review and bibliography cleanup.
- Finalize manuscript wording, limitations, acknowledgments, and submission materials.
- Perform a final reproducibility audit of benchmark scripts, SLURM workflows, analysis scripts, CSV results, and figures.
- Clean the local repository after preserving all final experimental files.
- Update README.md, ROADMAP.md, CHANGELOG.md, dependency documentation, and final weekly research logs to reflect the completed project.
- Review the final Git diff and create the final project cleanup commits.
- Synchronize the completed `kyle-hpc` branch with both the personal GitHub repository and the DREU-QIS-2026 organization repository.
- Complete the final DREU report and presentation materials.

---

# References

- Pittsburgh Supercomputing Center Bridges-2 User Guide
- ACCESS Documentation
- MPI Forum, MPI Standard
- Gropp et al., *Using MPI*
- Qiskit Documentation
- Qiskit Aer Documentation
- NVIDIA CUDA Documentation
- NVIDIA cuQuantum Documentation
- NinaPro DB2 Dataset Documentation
- qsim Documentation and Source Repository
- QuantumHPC Project Repository