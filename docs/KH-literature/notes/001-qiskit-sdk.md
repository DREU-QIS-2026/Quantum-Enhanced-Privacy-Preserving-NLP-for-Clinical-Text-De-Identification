# 001 – Quantum Computing with Qiskit

## IEEE Reference

[1] A. Javadi-Abhari *et al.*, "Quantum Computing with Qiskit," *arXiv preprint* arXiv:2405.08810, May 2024, doi: 10.48550/arXiv.2405.08810.

---

## Purpose

This is the official academic reference for the Qiskit SDK recommended by IBM for research publications. Useful in learning Qiskit for developing quantum circuits for the QuantumHPC project.

---

## Summary

The paper introduces the Qiskit Software Development Kit (SDK), explaining its design philosophy, software architecture, transpilation process, circuit representation, primitives, visualization tools, and extensibility. It also demonstrates an end-to-end quantum computing workflow using Qiskit and serves as the canonical citation for researchers using the SDK.

---

## Important Topics

- QuantumCircuit
- Transpiler
- Primitives
- Visualization
- SDK architecture
- Plugins
- Dynamic circuits

---

## Relevance

- Manual circuit construction
- Repository architecture
- Documentation
- Benchmarking framework
- Qiskit programming model

---

## Personal Notes

- Primarily in Python language, has interfaces and properties in place that allow great flexibility
- Agnositic to underlying hardware (very versatile, good when you may need to test on multiple different types of hardware) [Target class]
- Generally operates in a four-step workflow: circuit mapping (quantum encoding) -> transpilation (transformation to work with hardware) -> evalutation -> post-processing
- Includes multiple types of circuit visualization capabilities