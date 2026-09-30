# 002 – Qiskit Aer

## IEEE Reference

[2] Qiskit Development Team, Qiskit Aer Documentation, Qiskit Aer 0.17.1. Available: qiskit.github.io/qiskit-aer. Accessed: Jul. 15, 2026.

---

## Purpose

Qiskit Aer is the simulator used throughout the QuantumHPC project. Primarily used to test quantum circuits developed early in the project.

---

## Summary

Qiskit Aer provides high-performance classical simulation of quantum circuits, including statevector, density matrix, stabilizer, tensor-network, and GPU-accelerated simulation methods. It also supports realistic quantum noise models and serves as the primary backend for testing quantum algorithms without requiring access to physical quantum hardware.

---

## Important Topics

- AerSimulator
- Statevector
- GPU simulation
- Noise models
- Measurement
- Performance

---

## Relevance

- AerSimulator
- GPU benchmarking
- Runtime measurements
- Jetstream2
- Bridges-2

---

## Personal Notes

- class AerSimulator(configuration, properties, provider, target, backend_options)
- backend = AerSimulator
- Multiple simulation methods available, including "statevector" which is a statevector simulation which appears to be useful when dealing with noisy simulations, each shot randomly samples a circuit.
- Some simulation methods, including statevector, are suported on GPU (additionally density_matrix, unitary, and tensor_network [GPU only])
- Set device="GPU" to run a GPU simulation
- Many other GPU configurations and commands can be seen in the documentation, such as methods of checking available GPU(s) and their IDs if there are multiple involved.
- num_qubits teturns the number of qubits in the backend
- Also a variety of noise modeling is included in the documentation which may be beneficial for the future of the project