# Week 07

**Date:** August 24 – August 30, 2026

---

# Goals

- Transition from the validated HPC/MPI prototype toward the project's actual quantum-kernel workload.
- Coordinate with teammates on the quantum-kernel interface and prepare for integration with the HPC environment.
- Validate the existing manual, gate-by-gate ZZFeatureMap implementation against the project's quantum-kernel requirements.
- Begin preparing the project for secondary-dataset testing using the NinaPro DB2 EMG dataset.
- Prepare the project for the planned Jetson Nano execution and benchmarking test.
- Establish a reproducible local workload that can later be distributed through MPI and benchmarked across CPU/GPU resources.

---

# Approach & Implementation

Reviewed the current development state and coordinated the HPC work with the quantum-kernel interface provided by teammate Eddie. The interface uses feature vectors encoded into quantum states and computes kernel values from the overlap between encoded states. The current project configuration uses a single feature-map repetition, matching the configuration used by the teammate's implementation.

Validated the project's manual, gate-by-gate ZZFeatureMap implementation using a 4-qubit test case. The manual circuit was compared against the Qiskit reference feature-map implementation using state fidelity. The resulting fidelity was 1.000000000000, confirming that the manual implementation produces the same physical quantum state as the reference implementation for the tested configuration.

Implemented and tested a local quantum-kernel calculation using the manual feature map. For two 4-dimensional feature vectors, the kernel was calculated as the squared magnitude of the state overlap. The test confirmed the expected diagonal, symmetry, and range properties, including K(X,X) = 1 and K(X,Y) = K(Y,X).

Extended the validation to a 4x4 quantum kernel matrix. Each feature vector was encoded once and the unique pairwise kernel values were calculated and mirrored to form the symmetric matrix. The resulting matrix passed diagonal, symmetry, range, and positive-semidefinite checks. The eigenvalues were all positive for the tested matrix.

Refactored the validated kernel calculation into a reusable `simulation/quantum_kernel.py` component. The existing kernel-matrix test was updated to use this reusable component, and the refactored implementation reproduced the same kernel matrix and validation results.

Validated the existing Aer CPU backend path separately. A 4-qubit manual ZZFeatureMap was executed through the project's existing `BackendFactory` and Aer statevector backend. The resulting state had dimension 16 and normalization 1.000000000000, confirming that the existing Aer CPU statevector path is functional. This test also identified the need to explicitly request statevector saving when retrieving an Aer statevector result.

Acquired the NinaPro DB2 dataset for future secondary-dataset experiments and placed the data under the project's `data/ninapro/` directory. The dataset is excluded from version control through the project's `.gitignore` configuration so that the large dataset files are not committed to GitHub.

Prepared for the planned Jetson Nano test by creating a Jetson environment checklist and a small Qiskit/Aer smoke-test script. The checklist is intended to record the Jetson model, JetPack/L4T version, operating system, Python, CUDA, Qiskit/Aer, MPI/mpi4py, GPU status, and power/performance configuration before attempting project execution.

Maintained the existing project architecture rather than rewriting the current simulator or MPI prototype. The current work establishes a validated quantum-kernel workload that can be connected to the existing HPC/MPI infrastructure after the Jetson environment and backend compatibility are characterized.

---

# Results

- Confirmed compatibility between the project's manual gate-by-gate ZZFeatureMap and the Qiskit reference implementation for the tested 4-qubit configuration.
- Achieved state fidelity of 1.000000000000 between the manual and reference circuits.
- Successfully calculated individual quantum-kernel values from encoded state overlaps.
- Successfully generated a valid 4x4 quantum kernel matrix.
- Verified kernel diagonal, symmetry, range, and positive-semidefinite properties.
- Created a reusable `QuantumKernel` implementation without changing the validated numerical results.
- Confirmed that the existing Aer CPU backend can execute the manual circuit and return a normalized 4-qubit statevector.
- Added the NinaPro DB2 dataset locally while keeping the dataset excluded from Git tracking.
- Prepared the Jetson Nano environment checklist and smoke-test workflow for the upcoming hardware session.
- Established a validated local quantum-kernel workload suitable for the next stage of MPI/HPC integration and benchmarking.

---

# Next Steps

- Characterize the Jetson Nano software and hardware environment and determine Qiskit/Aer compatibility.
- Run the prepared Aer smoke test and, if supported, a small manual ZZFeatureMap/kernel workload on the Jetson Nano.
- Integrate the reusable quantum-kernel implementation with the project's existing Aer backend abstraction so CPU, GPU, and cuQuantum execution can be benchmarked consistently.
- Adapt the existing MPI prototype to distribute the validated quantum-kernel workload rather than independent demonstration simulations.
- Establish controlled MPI kernel-matrix scaling experiments and compare runtime, speedup, and parallel efficiency.
- Begin processing the NinaPro DB2 data into the feature representation required by the quantum-kernel interface.
- Coordinate further with teammates on the differential-privacy and quantum-kernel interfaces before connecting the complete pipeline.
- Continue toward CPU/GPU comparisons, qubit-scaling experiments, and the final performance dataset for the project paper.

---

# References

- A. Javadi-Abhari et al., “Quantum Computing with Qiskit,” *arXiv preprint* arXiv:2405.08810, May 2024, doi: 10.48550/arXiv.2405.08810.
- Qiskit Development Team, *Qiskit Aer Documentation*.
- V. Havlíček et al., “Supervised learning with quantum-enhanced feature spaces,” *Nature*, vol. 567, no. 7747, pp. 209–212, 2019, doi: 10.1038/s41586-019-0980-2.
- M. Atzori et al., “Electromyography data for non-invasive naturally-controlled robotic hand prostheses,” *Scientific Data*, vol. 1, 140053, 2014, doi: 10.1038/sdata.2014.53.
- Project GitHub Repository.