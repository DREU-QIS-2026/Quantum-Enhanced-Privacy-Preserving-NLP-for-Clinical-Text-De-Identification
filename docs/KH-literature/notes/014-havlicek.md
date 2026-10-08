# 014 – Supervised Learning with Quantum-Enhanced Feature Spaces

## IEEE Reference

[14] V. Havlíček, A. D. Córcoles, K. Temme, A. W. Harrow, A. Kandala, J. M. Chow, and J. M. Gambetta, "Supervised learning with quantum-enhanced feature spaces," *Nature*, vol. 567, no. 7747, pp. 209–212, 2019, doi: 10.1038/s41586-019-0980-2.

---

## Purpose

Foundational reference for the quantum feature-map and quantum-kernel approach used in the QuantumHPC project. Useful for understanding how classical data can be mapped into quantum feature spaces and compared using quantum-state overlaps.

---

## Summary

The paper presents two quantum machine-learning methods based on mapping classical data into a quantum feature space. One method uses a variational quantum classifier, while the second directly estimates a quantum kernel and uses the resulting kernel matrix with a classical support vector machine. The proposed feature map uses data-dependent phase operations and entangling interactions to create quantum states whose overlaps define the kernel.

---

## Important Topics

- Quantum feature spaces
- Quantum feature maps
- Quantum kernels
- State overlap / fidelity
- Support vector machines
- ZZ-style entangling interactions
- Quantum kernel estimation
- Classical simulation difficulty

---

## Relevance

- Theoretical basis for the QuantumHPC quantum kernel
- Definition of the kernel as a squared quantum-state overlap
- Manual quantum feature-map implementation
- Interpretation of the resulting kernel matrix
- Connection between classical feature vectors and quantum states

---

## Notes

- Classical data is mapped into a quantum state using a feature-map circuit
- Quantum kernel is defined as K(x,z) = |<Phi(x)|Phi(z)>|^2
- Kernel values can be calculated for every pair of data points to construct a full kernel matrix
- A classical SVM can use the quantum-generated kernel after the matrix has been calculated
- The feature map includes Hadamard gates, data-dependent phase operations, and two-qubit interactions
- Two-qubit diagonal interactions can be constructed using CNOT and phase/Z-style gates
- A possible quantum advantage depends on the selected feature map producing a kernel that is difficult to estimate classically
- The actual experimental data in this paper was artificially generated, so the paper mainly provides the theoretical and circuit basis for my implementation rather than a direct biomedical-data comparison