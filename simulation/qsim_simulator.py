"""
qsim CUDA simulator adapter.

Provides statevector simulation of the project's manual
ZZFeatureMap using Cirq/qsim CUDA on the local NVIDIA GPU.
"""

import numpy as np
import cirq
import qsimcirq


class QSimSimulator:
    """
    Adapter between the project's Qiskit circuits and qsim CUDA.

    The project continues to use its manual Qiskit circuit as the
    source of truth. This class translates the supported primitive
    gates to an equivalent Cirq circuit before qsim execution.
    """

    def __init__(self):
        self.options = qsimcirq.QSimOptions(
            use_gpu=True,
            gpu_mode=0,
        )

        self.simulator = qsimcirq.QSimSimulator(
            self.options
        )

    def _to_cirq(self, circuit):
        """
        Translate a bound Qiskit circuit into an equivalent
        Cirq circuit.

        Supported gates:
            H
            P
            CX
        """

        num_qubits = circuit.num_qubits

        qubits = cirq.LineQubit.range(
            num_qubits
        )

        cirq_circuit = cirq.Circuit()

        for instruction in circuit.data:

            operation = instruction.operation
            qiskit_qubits = instruction.qubits

            name = operation.name

            # ------------------------------------------------
            # Hadamard
            # ------------------------------------------------

            if name == "h":

                q = qiskit_qubits[0]._index

                cirq_circuit.append(
                    cirq.H(
                        qubits[q]
                    )
                )

            # ------------------------------------------------
            # Phase gate P(theta)
            # ------------------------------------------------

            elif name == "p":

                q = qiskit_qubits[0]._index

                theta = float(
                    operation.params[0]
                )

                cirq_circuit.append(
                    cirq.ZPowGate(
                        exponent=theta / np.pi
                    ).on(
                        qubits[q]
                    )
                )

            # ------------------------------------------------
            # Controlled-X / CNOT
            # ------------------------------------------------

            elif name == "cx":

                control = qiskit_qubits[0]._index
                target = qiskit_qubits[1]._index

                cirq_circuit.append(
                    cirq.CNOT(
                        qubits[control],
                        qubits[target],
                    )
                )

            # ------------------------------------------------
            # Ignore barriers
            # ------------------------------------------------

            elif name == "barrier":

                continue

            else:

                raise ValueError(
                    f"Unsupported Qiskit gate: {name}"
                )

        return cirq_circuit

    @staticmethod
    def _reorder_statevector(
        statevector,
        num_qubits,
    ):
        """
        Convert qsim/Cirq computational-basis ordering
        to Qiskit's ordering.
        """

        reordered = np.zeros_like(
            statevector
        )

        for index in range(
            len(statevector)
        ):

            reversed_index = int(
                format(
                    index,
                    f"0{num_qubits}b",
                )[::-1],
                2,
            )

            reordered[index] = statevector[
                reversed_index
            ]

        return reordered

    def statevector(
        self,
        circuit,
        feature_vector=None,
    ):
        """
        Simulate a Qiskit circuit using qsim CUDA.

        If feature_vector is supplied, circuit parameters are
        bound before translation.
        """

        if feature_vector is not None:

            circuit = circuit.assign_parameters(
                feature_vector
            )

        cirq_circuit = self._to_cirq(
            circuit
        )

        result = self.simulator.simulate(
            cirq_circuit
        )

        statevector = np.asarray(
            result.final_state_vector,
            dtype=np.complex64,
        )

        return self._reorder_statevector(
            statevector,
            circuit.num_qubits,
        )