# Week 09

**Date:** September 7 – September 13, 2026

---

# Goals

- Complete the controlled NinaPro real-data quantum-kernel benchmark.
- Characterize the effect of depolarizing noise on the NinaPro quantum workload.
- Develop and validate strong and weak MPI scaling experiments using the NinaPro workload.
- Prepare the final Bridges-2 benchmarking workflow and Slurm submission.
- Validate the final benchmark environment and synchronize the project repository across local and HPC systems.
- Begin transitioning the completed experimental work toward the final DREU paper and presentation.

---

# Approach & Implementation

Continued the project from the local GPU simulation work completed during Week 8 by transitioning the benchmarking workflow to the NinaPro DB2 real-data workload. The selected workload uses the S1/E1/A1 NinaPro dataset and extracts 102 gesture/repetition windows from 12 EMG channels. Each window is converted into 36 classical EMG features using mean absolute value, root mean square, and zero-crossing features. The features are standardized, reduced to 8 dimensions using PCA, scaled to the quantum-feature range, and processed using the project's manually constructed ZZFeatureMap and quantum-kernel implementation.

Completed local validation of the full NinaPro quantum-kernel matrix using both the CPU implementation and the CUDA-enabled qsim GPU implementation. The resulting 102 × 102 kernel matrices were numerically consistent, providing validation that the GPU backend could reproduce the project's quantum-kernel workload before moving to HPC benchmarking.

Completed additional local performance experiments using the NinaPro workload and evaluated runtime as the number of qubits increased. The qubit-scaling experiments were used to establish local CPU and GPU performance baselines for comparison with the planned Bridges-2 results.

Developed a separate noise-characterization benchmark using Qiskit Aer on the CPU. The experiment evaluates depolarizing noise at multiple noise strengths and compares the resulting noisy density matrix against the corresponding ideal quantum state. The noise experiment was expanded to use five depolarizing-noise strengths, p = {0.001, 0.005, 0.010, 0.020, 0.050}, providing a broader characterization of the effect of increasing simulation noise on the NinaPro quantum workload.

Developed the final MPI benchmark around the NinaPro quantum-kernel matrix calculation. The strong-scaling experiment maintains the actual 102-sample workload while increasing the number of MPI processes from 1 to 2, 4, and 8. The weak-scaling experiment increases the computational workload with the number of processes so that the number of kernel-pair calculations assigned to each process remains approximately constant. The weak-scaling workloads use 102, 144, 204, and 289 samples for 1, 2, 4, and 8 processes, respectively. The additional weak-scaling workloads are deterministic extensions of the real NinaPro feature set and are treated as computational scaling workloads rather than additional biological observations.

Created analysis scripts for the NinaPro noise and MPI experiments to generate runtime, speedup, and efficiency measurements and corresponding plots from the collected benchmark data. The final Slurm workflow was also prepared to execute the noise characterization, strong MPI scaling, weak MPI scaling, and analysis steps sequentially within a single Bridges-2 batch job.

Maintained the project repository throughout the final benchmarking preparation. The final benchmark source files, analysis scripts, and Slurm workflow were committed to Git and synchronized between the local repository, personal GitHub fork, project organization repository, and Bridges-2. The NinaPro dataset was also transferred to Bridges-2 and verified on the HPC filesystem.

Validated the Bridges-2 software environment before final submission. The project environment was confirmed to use Python 3.12.13, Qiskit 2.5.0, Qiskit Aer 0.17.2, and mpi4py 4.1.2. The final benchmark and analysis scripts successfully passed Python compilation on Bridges-2, and the final Slurm script was verified as a valid shell script.

Attempted to submit the final Bridges-2 benchmark after completing the environment and repository checks. The submission was rejected because the temporary workshop/student allocation used for Bridges-2 access was no longer active for Slurm job submission. The allocation remained visible through the project-accounting interface, but Slurm did not report an active account association for the user. The final HPC measurements therefore could not be completed during this week.

Discussed the remaining DREU project scope with the PI. The PI confirmed that the DREU work could be completed independently of the agentic MPI implementation being developed with Sofia. The agentic MPI architecture can therefore be treated as a future extension rather than a requirement for the current DREU submission.

---

# Results

- Completed the NinaPro DB2 S1/E1/A1 real-data quantum-kernel workload.
- Validated the full NinaPro quantum-kernel matrix between the local CPU implementation and CUDA-enabled qsim GPU implementation.
- Established local CPU and RTX 2080 SUPER GPU performance baselines for the NinaPro workload.
- Completed local qubit-scaling experiments across increasing numbers of qubits.
- Developed the final NinaPro depolarizing-noise characterization experiment.
- Expanded the noise experiment to five depolarizing-noise strengths.
- Developed strong MPI scaling experiments for 1, 2, 4, and 8 processes.
- Developed weak MPI scaling experiments for 1, 2, 4, and 8 processes.
- Completed the analysis scripts and final Slurm workflow for the Bridges-2 experiments.
- Transferred and verified the NinaPro dataset on Bridges-2.
- Synchronized the finalized benchmark code across the local repository, GitHub repositories, and Bridges-2.
- Verified the required Python, Qiskit, Qiskit Aer, and mpi4py environment on Bridges-2.
- Successfully compiled the final benchmark and analysis scripts on Bridges-2.
- Prepared the complete final Bridges-2 benchmark as a single batch workflow.
- Attempted the final Bridges-2 submission but was blocked by the expiration of the temporary workshop/student allocation and the resulting lack of an active Slurm account association.
- Confirmed with the PI that the DREU submission can be completed independently of the agentic MPI implementation.

---

# Next Steps

- Obtain continued ACCESS/Bridges-2 access through the appropriate allocation process.
- Run the final NinaPro depolarizing-noise characterization on Bridges-2.
- Run the strong MPI scaling benchmark using 1, 2, 4, and 8 processes.
- Run the weak MPI scaling benchmark using 1, 2, 4, and 8 processes.
- Analyze the final Bridges-2 runtime, speedup, and parallel-efficiency results.
- Compare local CPU, local GPU, and Bridges-2 performance.
- Integrate the completed benchmark results into the final DREU paper.
- Update the final project presentation with the completed experimental results.
- Clean and refactor the project repository after the final experimental results are collected.
- Document the agentic MPI architecture as potential future work outside the current DREU implementation scope.

---

# References

- Qiskit Documentation
- Qiskit Aer Documentation
- qsim Documentation and Source Repository
- NVIDIA CUDA Documentation
- MPI Forum Documentation
- Open MPI Documentation
- Bridges-2 User Documentation
- ACCESS Allocations Documentation
- NinaPro Database Documentation
- Project GitHub Repository