# Week 06

**Date:** August 17 – August 23, 2026

---

# Goals

- Begin the analysis and benchmarking phase following the completed Bridges-2 parameter sweep.
- Establish initial MPI scaling measurements and document the
resulting performance.
- Continue transferring completed technical work into the project
manuscript.
- Finalize the project workflow/flowchart and align it with the
remaining project schedule.
- Establish the GitHub organization and collaborative repository
structure for the team.
- Continue preparing the project for the Week 7--8 benchmarking,
dataset-extension, integration, and writing work.

---

# Approach & Implementation

Continued the HPC benchmarking work on Bridges-2 after completion of the
Week 5 parameter sweep. Developed and evaluated initial MPI scaling
experiments using 1, 2, and 4 MPI processes. Recorded runtime, speedup,
and parallel-efficiency measurements and generated plots for runtime
scaling, measured speedup versus ideal linear speedup, and parallel
efficiency.

Reviewed the existing Bridges-2 CPU parameter-sweep results by epsilon
and simulation/noise condition. Generated/organized plots showing
simulation time as a function of epsilon for amplitude-damping,
depolarizing, and noiseless conditions. These results provide the CPU
baseline against which later MPI and GPU experiments can be compared.

Continued manuscript preparation in Overleaf. Updated the Results
material, replaced remaining portions of the inherited template content,
resolved bibliography/citation issues, retained the required affiliation
placeholders, and incorporated the project flowchart into the
manuscript/presentation workflow. The manuscript now reflects the
project's actual HPC/Qiskit work rather than the original template.

Established the team's GitHub organization and moved the project
repository into the organization. Team contributors were migrated
successfully, and organization/repository collaboration settings were
configured. A personal copy of the repository was also created for
independent reference and verified against the organization repository.
The personal copy contains the same five branches and matching commit
hashes as the organization repository at the synchronization point.

Continued documenting the project architecture and the transition from
the initial CPU/HPC baseline toward MPI scaling, GPU benchmarking,
cross-resource testing, and the planned PhysioNet/NinaPro EMG extension.
---

# Results

- Completed initial MPI scaling measurements using 1, 2, and 4
processes.
- Measured approximately 1.0×, 1.74×, and 2.61× speedup for 1, 2, and 4 processes, respectively.
- Measured approximately 100%, 87%, and 65% parallel efficiency for 1, 2, and 4 processes.
- Demonstrated that increasing MPI process count reduced measured
wall-clock runtime while falling below ideal linear scaling.
- Generated runtime-scaling, speedup, and parallel-efficiency figures for the manuscript and project documentation.
- Continued analysis of the completed epsilon/noise-condition CPU
baseline.
- Updated the manuscript so that the Results and subsequent sections reflect the current project rather than the inherited template.
- Resolved manuscript citation/bibliography issues and retained
affiliation placeholders for later completion.
- Completed and incorporated the project flowchart.
- Established the DREU-QIS-2026 GitHub organization and moved the
project repository into it.
- Verified the personal repository as a Git-level mirror of the
organization repository at the current synchronization point.
- Maintained the project as a reproducible combination of code, experiment results, plots, documentation, and manuscript materials.

---

# Next Steps

- Extend MPI scaling beyond four processes and test larger workloads.
- Distinguish strong-scaling and weak-scaling experiments where
appropriate.
- Begin GPU execution on Bridges-2 and compare CPU, MPI-CPU, and GPU performance.
- Investigate how the MPI workload distribution should connect to the actual quantum-kernel computation rather than only the current
benchmark workload.
- Begin the planned PhysioNet/NinaPro DB2 EMG quantum-kernel
extension.
- Coordinate the HPC results with the quantum-kernel and
differential-privacy tracks.
- Continue filling the literature review with source-specific notes and connect the references directly to methodology and
implementation decisions.
- Continue expanding the Overleaf manuscript toward a complete Week 6--8 paper draft.
- Prepare for the Week 8 technical/research review and subsequent
final benchmarking, replication, and submission work.

---

# References

- Qiskit Documentation
- Qiskit Aer Documentation
- MPI Forum Documentation
- LLNL MPI Tutorial
- Bridges-2 User Documentation
- Brown et al., "Multi-GPU Quantum Circuit Simulation and the Impact of Network Performance"
- Project GitHub Repository
- Updated DREU-QIS 2026 Project Overview