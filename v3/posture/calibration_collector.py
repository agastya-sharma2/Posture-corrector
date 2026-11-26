import math
from typing import Optional, Sequence

from posture.calibration import Calibration

class CalibrationCollector:
    def __init__(self, target_samples=10, min_samples=5):
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")

        if target_samples < min_samples:
            raise ValueError(
                "target_samples must be at least min_samples"
            )

        self.target_samples = target_samples
        self.min_samples = min_samples
        self.samples = []

    def add_sample(self, angles: Optional[Sequence[Optional[float]]]):
        if angles is None or isinstance(angles, (str, bytes)):
            return False

        try:
            if len(angles) != 3:
                return False

            if any(isinstance(angle, bool) for angle in angles):
                return False

            angles = tuple(float(angle) for angle in angles)

        except (TypeError, ValueError, OverflowError):
            return False

        if not all(
            math.isfinite(angle) and 0 <= angle <= 180
            for angle in angles
        ):
            return False

        if self.is_complete():
            return False

        self.samples.append(angles)
        return True

    def is_complete(self):
        return len(self.samples) >= self.target_samples

    def get_calibration(self) -> Calibration:
        if len(self.samples) < self.min_samples:
            raise ValueError(
                f"Not enough valid samples: got {len(self.samples)}, "
                f"need at least {self.min_samples}"
            )

        return Calibration.from_samples(
            self.samples,
            min_samples=self.min_samples
        )

    def reset(self):
        self.samples.clear()