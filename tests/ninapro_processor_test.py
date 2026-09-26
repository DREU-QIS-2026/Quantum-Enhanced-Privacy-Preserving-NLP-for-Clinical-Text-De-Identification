import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

import numpy as np

from datasets.ninapro import NinaProProcessor


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASET = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "ninapro"
    / "DB2"
    / "s1"
    / "S1_E1_A1.mat"
)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("NinaPro DB2 Processor Validation")
    print("=" * 60)

    print(f"\nDataset:")
    print(DATASET)

    processor = NinaProProcessor(
        DATASET
    )

    print("\nProcessor configuration")
    print("-" * 60)

    info = processor.info()

    for key, value in info.items():
        print(
            f"{key:20}: {value}"
        )

    print("\nProcessing NinaPro data...")

    features, labels, metadata = (
        processor.process()
    )

    print("\nOutput")
    print("-" * 60)

    print(
        f"Feature matrix shape : "
        f"{features.shape}"
    )

    print(
        f"Labels shape         : "
        f"{labels.shape}"
    )

    print(
        f"Metadata entries     : "
        f"{len(metadata)}"
    )

    print(
        f"Feature minimum      : "
        f"{features.min():.6f}"
    )

    print(
        f"Feature maximum      : "
        f"{features.max():.6f}"
    )

    print(
        f"Feature mean         : "
        f"{features.mean():.6f}"
    )

    print(
        f"Feature std          : "
        f"{features.std():.6f}"
    )

    print(
        f"Unique labels        : "
        f"{np.unique(labels)}"
    )

    # --------------------------------------------------------
    # Label counts
    # --------------------------------------------------------

    print("\nLabel counts")
    print("-" * 60)

    for label in np.unique(labels):

        count = np.sum(
            labels == label
        )

        print(
            f"Gesture {label:2d}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # First metadata entries
    # --------------------------------------------------------

    print("\nFirst 3 metadata entries")
    print("-" * 60)

    for entry in metadata[:3]:
        print(entry)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    expected_samples = 17 * 6

    shape_valid = (
        features.shape
        == (
            expected_samples,
            8,
        )
    )

    labels_valid = (
        len(labels)
        == expected_samples
    )

    metadata_valid = (
        len(metadata)
        == expected_samples
    )

    label_set_valid = np.array_equal(
        np.unique(labels),
        np.arange(1, 18),
    )

    counts_valid = all(
        np.sum(labels == label) == 6
        for label in range(1, 18)
    )

    finite_valid = np.all(
        np.isfinite(features)
    )

    range_valid = (
        np.all(features >= -1e-9)
        and np.all(
            features <= np.pi + 1e-9
        )
    )

    print("\nValidation")
    print("-" * 60)

    print(
        f"Shape = (102, 8)       : "
        f"{shape_valid}"
    )

    print(
        f"102 labels             : "
        f"{labels_valid}"
    )

    print(
        f"102 metadata entries   : "
        f"{metadata_valid}"
    )

    print(
        f"Gestures 1-17          : "
        f"{label_set_valid}"
    )

    print(
        f"6 windows per gesture  : "
        f"{counts_valid}"
    )

    print(
        f"All values finite      : "
        f"{finite_valid}"
    )

    print(
        f"Features in [0, pi]    : "
        f"{range_valid}"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    if all([
        shape_valid,
        labels_valid,
        metadata_valid,
        label_set_valid,
        counts_valid,
        finite_valid,
        range_valid,
    ]):

        print(
            "\nPASS: NinaPro processor produced "
            "the expected 102 x 8 quantum feature matrix."
        )

    else:

        print(
            "\nFAIL: NinaPro processor validation failed."
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()