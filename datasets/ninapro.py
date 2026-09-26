"""
NinaPro DB2 EMG preprocessing.

Loads a NinaPro DB2 MATLAB file and converts selected gesture
segments into fixed-size quantum feature vectors.

Pipeline:

    .mat file
        ↓
    12-channel EMG
        ↓
    512-sample windows
        ↓
    MAV + RMS + ZC per channel
        ↓
    36 classical features
        ↓
    StandardScaler
        ↓
    PCA → 8 dimensions
        ↓
    MinMaxScaler → [0, pi]
        ↓
    8 quantum features
"""

from pathlib import Path

import numpy as np
import scipy.io as sio

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class NinaProProcessor:
    """
    Process one NinaPro DB2 exercise file.

    The current benchmark configuration uses:

        - 12 EMG channels
        - gesture labels 1-17
        - repetitions 1-6
        - one 512-sample window per gesture/repetition
        - MAV, RMS, and zero crossings
        - PCA to 8 dimensions
        - final scaling to [0, pi]
    """

    SAMPLING_RATE = 2000
    WINDOW_SIZE = 512

    NUM_CHANNELS = 12

    GESTURE_LABELS = tuple(range(1, 18))
    REPETITIONS = tuple(range(1, 7))

    NUM_RAW_FEATURES = 36
    NUM_QUANTUM_FEATURES = 8

    def __init__(
        self,
        filepath,
        window_size=WINDOW_SIZE,
        n_components=NUM_QUANTUM_FEATURES,
    ):
        self.filepath = Path(filepath)
        self.window_size = window_size
        self.n_components = n_components

        self.scaler = StandardScaler()
        self.pca = PCA(
            n_components=n_components,
            random_state=42,
        )
        self.quantum_scaler = MinMaxScaler(
            feature_range=(0.0, np.pi)
        )

    # --------------------------------------------------------
    # Loading
    # --------------------------------------------------------

    def load(self):
        """
        Load the NinaPro MATLAB file.
        """

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"NinaPro file not found: {self.filepath}"
            )

        data = sio.loadmat(self.filepath)

        required = [
            "emg",
            "restimulus",
            "rerepetition",
        ]

        missing = [
            name
            for name in required
            if name not in data
        ]

        if missing:
            raise KeyError(
                "Missing required NinaPro variables: "
                + ", ".join(missing)
            )

        emg = np.asarray(
            data["emg"],
            dtype=np.float32,
        )

        restimulus = np.asarray(
            data["restimulus"]
        ).ravel()

        rerepetition = np.asarray(
            data["rerepetition"]
        ).ravel()

        if emg.ndim != 2:
            raise ValueError(
                f"Expected 2D EMG array, got shape {emg.shape}"
            )

        if emg.shape[1] != self.NUM_CHANNELS:
            raise ValueError(
                f"Expected {self.NUM_CHANNELS} EMG channels, "
                f"got {emg.shape[1]}"
            )

        if len(restimulus) != len(emg):
            raise ValueError(
                "restimulus length does not match EMG samples"
            )

        if len(rerepetition) != len(emg):
            raise ValueError(
                "rerepetition length does not match EMG samples"
            )

        return emg, restimulus, rerepetition

    # --------------------------------------------------------
    # Segment detection
    # --------------------------------------------------------

    @staticmethod
    def _contiguous_runs(mask):
        """
        Return contiguous [start, end) runs where mask is True.
        """

        indices = np.flatnonzero(mask)

        if len(indices) == 0:
            return []

        split_points = np.flatnonzero(
            np.diff(indices) != 1
        ) + 1

        groups = np.split(
            indices,
            split_points,
        )

        return [
            (int(group[0]), int(group[-1]) + 1)
            for group in groups
            if len(group) > 0
        ]

    def _find_segment(
        self,
        restimulus,
        rerepetition,
        gesture,
        repetition,
    ):
        """
        Find the longest contiguous segment corresponding
        to one gesture/repetition pair.
        """

        mask = (
            (restimulus == gesture)
            & (rerepetition == repetition)
        )

        runs = self._contiguous_runs(mask)

        if not runs:
            raise ValueError(
                f"No segment found for gesture={gesture}, "
                f"repetition={repetition}"
            )

        start, end = max(
            runs,
            key=lambda pair: pair[1] - pair[0]
        )

        length = end - start

        if length < self.window_size:
            raise ValueError(
                f"Segment too short for gesture={gesture}, "
                f"repetition={repetition}: "
                f"{length} samples"
            )

        return start, end

    # --------------------------------------------------------
    # Window extraction
    # --------------------------------------------------------

    def _extract_center_window(
        self,
        emg,
        start,
        end,
    ):
        """
        Extract one deterministic center window.
        """

        segment_length = end - start

        window_start = (
            start
            + (segment_length - self.window_size) // 2
        )

        window_end = (
            window_start
            + self.window_size
        )

        return emg[
            window_start:window_end
        ]

    # --------------------------------------------------------
    # EMG features
    # --------------------------------------------------------

    @staticmethod
    def _mav(signal):
        """Mean Absolute Value."""

        return np.mean(
            np.abs(signal)
        )

    @staticmethod
    def _rms(signal):
        """Root Mean Square."""

        return np.sqrt(
            np.mean(
                signal ** 2
            )
        )

    @staticmethod
    def _zero_crossings(signal):
        """
        Count sign changes between consecutive samples.
        Exact zeros do not count as crossings.
        """

        return np.count_nonzero(
            signal[:-1] * signal[1:] < 0
        )

    def _extract_features(self, window):
        """
        Extract MAV, RMS, and ZC for all 12 channels.

        Feature ordering:

            channel 1: MAV, RMS, ZC
            channel 2: MAV, RMS, ZC
            ...
            channel 12: MAV, RMS, ZC
        """

        features = []

        for channel in range(
            self.NUM_CHANNELS
        ):

            signal = window[:, channel]

            features.extend([
                self._mav(signal),
                self._rms(signal),
                self._zero_crossings(signal),
            ])

        return np.asarray(
            features,
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # Main processing
    # --------------------------------------------------------

    def process(self):
        """
        Create the final 8-dimensional quantum feature matrix.

        Returns
        -------
        quantum_features : ndarray
            Shape: (102, 8)

        labels : ndarray
            Gesture label for each feature vector.

        metadata : list of dict
            Information about each selected window.
        """

        emg, restimulus, rerepetition = self.load()

        raw_features = []
        labels = []
        metadata = []

        # ----------------------------------------------------
        # One window per gesture/repetition
        # ----------------------------------------------------

        for gesture in self.GESTURE_LABELS:

            for repetition in self.REPETITIONS:

                start, end = self._find_segment(
                    restimulus,
                    rerepetition,
                    gesture,
                    repetition,
                )

                window = self._extract_center_window(
                    emg,
                    start,
                    end,
                )

                features = self._extract_features(
                    window
                )

                raw_features.append(
                    features
                )

                labels.append(
                    gesture
                )

                metadata.append({
                    "gesture": gesture,
                    "repetition": repetition,
                    "segment_start": start,
                    "segment_end": end,
                    "window_start": (
                        start
                        + (end - start - self.window_size) // 2
                    ),
                    "window_size": self.window_size,
                })

        raw_features = np.asarray(
            raw_features,
            dtype=np.float64,
        )

        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        # ----------------------------------------------------
        # Standardize raw features
        # ----------------------------------------------------

        standardized = self.scaler.fit_transform(
            raw_features
        )

        # ----------------------------------------------------
        # PCA → 8 dimensions
        # ----------------------------------------------------

        reduced = self.pca.fit_transform(
            standardized
        )

        # ----------------------------------------------------
        # Scale quantum features to [0, pi]
        # ----------------------------------------------------

        quantum_features = (
            self.quantum_scaler.fit_transform(
                reduced
            )
        )

        return (
            quantum_features,
            labels,
            metadata,
        )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    def info(self):
        """
        Return a summary of the processor configuration.
        """

        return {
            "file": str(self.filepath),
            "sampling_rate_hz": self.SAMPLING_RATE,
            "emg_channels": self.NUM_CHANNELS,
            "window_size": self.window_size,
            "gesture_classes": len(
                self.GESTURE_LABELS
            ),
            "repetitions": len(
                self.REPETITIONS
            ),
            "raw_features": self.NUM_RAW_FEATURES,
            "quantum_features": self.n_components,
        }