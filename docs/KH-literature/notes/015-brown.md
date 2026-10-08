# 015 – Multi-GPU Quantum Circuit Simulation and the Impact of Network Performance

## IEEE Reference

[15] W. M. Brown, A. Ramesh, T. Lubinski, T. Nguyen, and D. E. Bernal Neira, "Multi-GPU Quantum Circuit Simulation and the Impact of Network Performance," *arXiv preprint* arXiv:2511.14664, Mar. 2026.

---

## Purpose

Primary HPC literature reference supplied by my mentor for understanding and implementing distributed and GPU-based quantum-simulation benchmarking. Useful for comparing my MPI and GPU experiments with a larger multi-GPU quantum-simulation study.

---

## Summary

The paper evaluates high-performance quantum-circuit simulation across multiple generations of NVIDIA GPUs and HPC interconnects. MPI support is added to the QED-C Application-Oriented Benchmark suite to enable distributed multi-GPU benchmarking. The study evaluates both strong and weak scaling and shows that communication between GPUs can become a major performance bottleneck as state-vector simulations are distributed across multiple GPUs and nodes.

---

## Important Topics

- GPU quantum-circuit simulation
- MPI
- Multi-GPU simulation
- Strong scaling
- Weak scaling
- State-vector simulation
- Network/interconnect performance
- Communication overhead
- CUDA-aware MPI
- Warm-up runs
- Synchronization before timing

---

## Relevance

- GPU and HPC quantum-simulation benchmarking
- MPI scaling methodology
- Strong- and weak-scaling experiments
- Comparison with the QuantumHPC MPI implementation
- Runtime and parallel-efficiency analysis
- Benchmark warm-up and synchronization methodology
- Possible future multi-GPU extensions

---

## Notes

- Classical state-vector simulation becomes much more expensive as qubit count increases because memory and computational requirements grow exponentially
- GPU acceleration can make quantum simulation significantly faster, but communication can become a bottleneck when a simulation is divided across multiple GPUs
- The paper adds MPI support to the QED-C quantum benchmark suite
- Uses both strong-scaling and weak-scaling experiments
- QPE is used as a weak-scaling benchmark and a fixed 33-qubit TFIM workload is used for strong scaling
- Random circuits are also tested as a more irregular workload
- The first circuit execution is excluded as a warm-up before timing
- MPI processes are synchronized with a barrier before timing so startup differences do not affect the benchmark
- Uses mpi4py in the Python-based QED-C benchmarks
- Their main MPI approach distributes a single state vector across multiple GPUs
- My QuantumHPC MPI implementation is different because I distribute independent quantum-kernel calculations across MPI ranks rather than distributing one state vector
- Their results show that faster GPU hardware alone does not determine total performance; the network/interconnect between GPUs can have a major effect on scaling