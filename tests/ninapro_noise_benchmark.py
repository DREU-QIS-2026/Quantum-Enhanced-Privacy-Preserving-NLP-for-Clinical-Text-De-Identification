"""
NinaPro real-data noise validation.

Runs the project's manual ZZFeatureMap on the 102-sample
NinaPro workload and evaluates the effect of depolarizing
noise on the resulting quantum states.

The noiseless state is generated with the project's existing
Qiskit statevector implementation.

Noisy states are generated with Qiskit Aer density-matrix
simulation using the project's NoiseModelFactory.

Noise conditions:
    - depolarizing p=0.005
    - depolarizing p=0.010
    - depolarizing p=0.020

For each NinaPro sample, the noisy-state fidelity relative
to the ideal state is recorded.

Raw per-sample results:
    results/logs/ninapro_noise_samples.csv

Summary results:
    results/logs/ninapro_noise_summary.csv
"""

from pathlib import Path
import csv
import sys
import time

import numpy as np

from qiskit import transpile
from qiskit_aer import AerSimulator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.ninapro import NinaProProcessor
from circuits.feature_maps import FeatureMapBuilder
from experiment.aer_noise_factory import NoiseModelFactory
from simulation.quantum_kernel import QuantumKernel


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASET = (
    PROJECT_ROOT
    / "data"
    / "ninapro"
    / "DB2"
    / "s1"
    / "S1_E1_A1.mat"
)

NOISE_STRENGTHS = [
    0.001,
    0.005,
    0.010,
    0.020,
    0.050,
]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "logs"
)

SAMPLE_RESULTS_FILE = (
    RESULTS_DIR
    / "ninapro_noise_samples.csv"
)

SUMMARY_RESULTS_FILE = (
    RESULTS_DIR
    / "ninapro_noise_summary.csv"
)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def noisy_density_matrix(
    circuit,
    feature_vector,
    noise_model,
    backend,
):
    """
    Simulate one noisy circuit and return its density matrix.
    """

    bound = circuit.assign_parameters(
        feature_vector
    )

    noisy_circuit = bound.copy()

    noisy_circuit.save_density_matrix()

    compiled = transpile(
        noisy_circuit,
        backend,
    )

    result = backend.run(
        compiled,
        noise_model=noise_model,
    ).result()

    return result.data(0)["density_matrix"]


def pure_state_fidelity(
    ideal_state,
    density_matrix,
):
    """
    Fidelity between an ideal pure state |psi> and a
    noisy density matrix rho.

    For a pure reference state:

        F = <psi|rho|psi>
    """

    psi = np.asarray(
        ideal_state,
        dtype=complex,
    )

    rho = np.asarray(
        density_matrix,
        dtype=complex,
    )

    fidelity = np.vdot(
        psi,
        rho @ psi,
    )

    fidelity = float(
        np.real_if_close(fidelity)
    )

    return float(
        np.clip(fidelity, 0.0, 1.0)
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("NinaPro Real-Data Noise Validation")
    print("=" * 70)

    print("\nDataset")
    print("-" * 70)
    print(DATASET)

    # --------------------------------------------------------
    # Process NinaPro
    # --------------------------------------------------------

    processor = NinaProProcessor(
        DATASET
    )

    features, labels, metadata = (
        processor.process()
    )

    num_samples = features.shape[0]
    num_features = features.shape[1]

    print("\nWorkload")
    print("-" * 70)
    print(f"Samples       : {num_samples}")
    print(f"Quantum features: {num_features}")
    print("Qubits        : 8")
    print("Feature map   : Manual ZZFeatureMap")
    print("Repetitions   : 1")

    # --------------------------------------------------------
    # Build manual circuit
    # --------------------------------------------------------

    builder = FeatureMapBuilder(
        reps=1,
        entanglement="linear",
    )

    circuit = builder.build_manual(
        num_features
    )

    print(
        f"Circuit depth : {circuit.depth()}"
    )

    # --------------------------------------------------------
    # Generate ideal states
    # --------------------------------------------------------

    print("\nGenerating ideal states...")
    print("-" * 70)

    ideal_kernel = QuantumKernel(
        reps=1,
        backend="cpu",
    )

    ideal_states = []

    ideal_start = time.perf_counter()

    for feature_vector in features:

        state = ideal_kernel.encode(
            feature_vector
        )

        ideal_states.append(
            np.asarray(
                state.data,
                dtype=complex,
            )
        )

    ideal_elapsed = (
        time.perf_counter()
        - ideal_start
    )

    ideal_states = np.asarray(
        ideal_states,
        dtype=complex,
    )

    print(
        f"Ideal-state generation: "
        f"{ideal_elapsed:.6f} seconds"
    )

    print(
        f"State shape           : "
        f"{ideal_states.shape}"
    )

    # --------------------------------------------------------
    # Aer density-matrix backend
    # --------------------------------------------------------

    backend = AerSimulator(
        method="density_matrix",
        device="CPU",
        max_parallel_threads=1,
        max_parallel_experiments=1,
    )

    # --------------------------------------------------------
    # Run noise conditions
    # --------------------------------------------------------

    sample_rows = []
    summary_rows = []

    for noise_strength in NOISE_STRENGTHS:

        condition = "depolarizing"

        print("\n" + "=" * 70)
        print(
            f"DEPOLARIZING NOISE p={noise_strength:.3f}"
        )
        print("=" * 70)

        noise_model = NoiseModelFactory.create(
            condition=condition,
            noise_strength=noise_strength,
        )

        fidelities = []
        runtimes = []

        for index, feature_vector in enumerate(
            features
        ):

            start = time.perf_counter()

            density_matrix = (
                noisy_density_matrix(
                    circuit,
                    feature_vector,
                    noise_model,
                    backend,
                )
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            fidelity = pure_state_fidelity(
                ideal_states[index],
                density_matrix,
            )

            fidelities.append(
                fidelity
            )

            runtimes.append(
                elapsed
            )

            sample_rows.append({
                "sample_index": index,
                "gesture": int(labels[index]),
                "repetition": int(
                    metadata[index]["repetition"]
                ),
                "condition": condition,
                "noise_strength": noise_strength,
                "runtime_seconds": elapsed,
                "ideal_noisy_fidelity": fidelity,
            })

        fidelities = np.asarray(
            fidelities,
            dtype=float,
        )

        runtimes = np.asarray(
            runtimes,
            dtype=float,
        )

        summary = {
            "condition": condition,
            "noise_strength": noise_strength,
            "samples": num_samples,
            "mean_fidelity": np.mean(
                fidelities
            ),
            "std_fidelity": np.std(
                fidelities,
                ddof=1,
            ),
            "min_fidelity": np.min(
                fidelities
            ),
            "max_fidelity": np.max(
                fidelities
            ),
            "mean_runtime_seconds": np.mean(
                runtimes
            ),
            "std_runtime_seconds": np.std(
                runtimes,
                ddof=1,
            ),
            "total_runtime_seconds": np.sum(
                runtimes
            ),
        }

        summary_rows.append(
            summary
        )

        print(
            f"Mean fidelity : "
            f"{summary['mean_fidelity']:.8f}"
        )

        print(
            f"Std fidelity  : "
            f"{summary['std_fidelity']:.8f}"
        )

        print(
            f"Min fidelity  : "
            f"{summary['min_fidelity']:.8f}"
        )

        print(
            f"Max fidelity  : "
            f"{summary['max_fidelity']:.8f}"
        )

        print(
            f"Mean runtime  : "
            f"{summary['mean_runtime_seconds']:.6f} s"
        )

    # --------------------------------------------------------
    # Save per-sample results
    # --------------------------------------------------------

    with SAMPLE_RESULTS_FILE.open(
        "w",
        newline="",
    ) as file:

        fieldnames = [
            "sample_index",
            "gesture",
            "repetition",
            "condition",
            "noise_strength",
            "runtime_seconds",
            "ideal_noisy_fidelity",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            sample_rows
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    with SUMMARY_RESULTS_FILE.open(
        "w",
        newline="",
    ) as file:

        fieldnames = [
            "condition",
            "noise_strength",
            "samples",
            "mean_fidelity",
            "std_fidelity",
            "min_fidelity",
            "max_fidelity",
            "mean_runtime_seconds",
            "std_runtime_seconds",
            "total_runtime_seconds",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    all_valid = all(
        np.isfinite(
            row["ideal_noisy_fidelity"]
        )
        and
        -1e-10 <= row["ideal_noisy_fidelity"] <= 1.0 + 1e-10
        for row in sample_rows
    )

    print("\n" + "=" * 70)

    if all_valid:
        print(
            "PASS: All noisy NinaPro states "
            "produced valid fidelity values."
        )
    else:
        print(
            "FAIL: Invalid fidelity detected."
        )

    print("=" * 70)

    print(
        f"\nSample results saved to:\n"
        f"{SAMPLE_RESULTS_FILE}"
    )

    print(
        f"\nSummary results saved to:\n"
        f"{SUMMARY_RESULTS_FILE}"
    )

    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()