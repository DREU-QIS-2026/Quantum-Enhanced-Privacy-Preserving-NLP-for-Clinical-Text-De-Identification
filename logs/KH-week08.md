# Week 08

**Date:** August 31 – September 6, 2026

---

# Goals

- Troubleshoot the existing Jetson benchmarking workflow.
- Investigate GPU simulation options for the local NVIDIA RTX 2080 SUPER.
- Establish a functional GPU-enabled quantum simulator in the native Windows environment.
- Validate GPU simulator execution before beginning performance benchmarking.

---

# Approach & Implementation

Began the week by troubleshooting the existing `jetson_benchmark.py` workflow and reviewing the preliminary Jetson benchmarking results. The Jetson experiments were treated as preliminary testing rather than part of the final benchmarking workflow, allowing the project to focus on the local system and Bridges-2 environments for the remaining performance experiments.

Investigated GPU simulation support in Qiskit Aer and determined that the existing native Windows installation provided CPU execution but did not provide the required GPU execution path. Alternative quantum simulators were therefore evaluated for native Windows GPU support. A Qulacs GPU build was investigated using the local CUDA and Visual Studio environment, but compiler and CUDA build issues prevented the GPU version from being completed efficiently.

Switched to qsim and created a separate Python 3.11 environment for GPU simulator development. Built qsim from source using the native Visual Studio and CUDA toolchain. During the build process, resolved CMake generator configuration, CUDA/MSVC compiler flag issues, and Python library linking problems. The resulting installation successfully produced the qsim CUDA extension.

Validated the compiled CUDA extension by importing `qsimcirq.qsim_cuda` and executing a Bell-state circuit using qsim with GPU execution enabled through `QSimOptions`. The resulting statevector matched the expected Bell-state output.

Finally, used `nvidia-smi` to verify that the workload was actually interacting with the local NVIDIA GeForce RTX 2080 SUPER. During GPU execution, the RTX 2080 SUPER reported active GPU utilization and increased GPU memory usage, confirming that the GPU-enabled qsim workload was executing on the local GPU rather than only loading the CUDA library.

---

# Results

- Troubleshooting of the Jetson benchmarking workflow was completed.
- Confirmed that the native Windows Qiskit Aer installation did not provide the required GPU execution path.
- Evaluated alternative quantum simulators for native Windows GPU execution.
- Successfully built and installed qsim with CUDA support.
- Successfully loaded the compiled `qsim_cuda` extension.
- Successfully executed a Bell-state circuit using qsim GPU execution.
- Verified that the resulting statevector matched the expected Bell state.
- Confirmed active utilization of the NVIDIA GeForce RTX 2080 SUPER during GPU execution using `nvidia-smi`.
- Established a functional native Windows GPU simulation environment for the project's next benchmarking phase.

---

# Next Steps

- Develop the controlled quantum simulation benchmark.
- Benchmark the local CPU and RTX 2080 SUPER GPU using the same workload.
- Evaluate runtime as the number of qubits increases.
- Run the benchmark workload on Bridges-2.
- Evaluate MPI scaling and parallel efficiency.
- Compare CPU, GPU, and HPC performance results.

---

# References

- Qiskit Documentation
- Qiskit Aer Documentation
- qsim Documentation and Source Repository
- NVIDIA CUDA Documentation
- NVIDIA `nvidia-smi` Documentation