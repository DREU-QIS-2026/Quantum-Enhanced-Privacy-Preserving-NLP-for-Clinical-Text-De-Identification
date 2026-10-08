"""
Quantum kernel evaluation utilities.

Provides quantum-kernel calculations using the project's
manual ZZFeatureMap and Qiskit statevector simulation.
"""

import numpy as np

from circuits.feature_maps import FeatureMapBuilder
from simulation import QuantumSimulator


class QuantumKernel:
    """
    Computes a quantum kernel from feature vectors.

    The kernel is defined as:

        K(x, y) = |<psi(x) | psi(y)>|^2

    where psi(x) is the quantum state produced by the
    project's manual ZZFeatureMap.
    """

    def __init__(
        self,
        reps=1,
        entanglement="linear",
        backend="cpu",
    ):
        self.reps = reps
        self.entanglement = entanglement

        self.builder = FeatureMapBuilder(
            reps=reps,
            entanglement=entanglement,
        )

        if backend == "gpu":
            from simulation.qsim_simulator import QSimSimulator
            self.simulator = QSimSimulator()
        else:
            self.simulator = QuantumSimulator(
                backend=backend,
                condition="noiseless",
            )

    def encode(self, feature_vector):
        """
        Encode one feature vector into a quantum state.
        """

        circuit = self.builder.build_manual(
            len(feature_vector)
        )

        return self.simulator.statevector(
            circuit,
            feature_vector,
        )

    @staticmethod
    def evaluate(state_a, state_b):
        """
        Calculate the quantum kernel between two states.

        K(a,b) = |<psi(a)|psi(b)>|^2
        """

        overlap = np.vdot(
            state_a.data,
            state_b.data,
        )

        return float(abs(overlap) ** 2)

    def evaluate_vectors(
        self,
        feature_vector_a,
        feature_vector_b,
    ):
        """
        Encode two feature vectors and calculate their kernel.
        """

        state_a = self.encode(feature_vector_a)
        state_b = self.encode(feature_vector_b)

        return self.evaluate(
            state_a,
            state_b,
        )

    def matrix(self, feature_vectors):
        """
        Build a symmetric quantum kernel matrix.

        Each feature vector is encoded exactly once.
        Only the upper triangle is calculated directly.
        """

        feature_vectors = np.asarray(
            feature_vectors,
            dtype=float,
        )

        n_samples = len(feature_vectors)

        states = [
            self.encode(feature_vector)
            for feature_vector in feature_vectors
        ]

        kernel_matrix = np.zeros(
            (n_samples, n_samples),
            dtype=float,
        )

        for i in range(n_samples):

            for j in range(i, n_samples):

                value = self.evaluate(
                    states[i],
                    states[j],
                )

                kernel_matrix[i, j] = value
                kernel_matrix[j, i] = value

        return kernel_matrix