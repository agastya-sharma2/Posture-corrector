import math

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class Calibration:
    """Personal reference angles captured during calibration."""

    neck_reference: float
    arm_reference: float
    head_reference: float
    sample_count: int

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[Sequence[Optional[float]]],
        min_samples: int = 5,
    ) -> "Calibration":
        """
        Create calibration references from angle samples.

        Each sample must contain:
        (neck_angle, arm_angle, head_angle)

        Samples with missing, non-finite, or out-of-range angles
        are ignored.
        """
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")

        valid_samples = []

        for sample in samples:
            if sample is None or len(sample) != 3:
                continue

            try:
                angles = tuple(float(angle) for angle in sample)
            except (TypeError, ValueError):
                continue

            if not all(
                math.isfinite(angle) and 0 <= angle <= 180
                for angle in angles
            ):
                continue

            valid_samples.append(angles)

        if len(valid_samples) < min_samples:
            raise ValueError(
                f"Not enough valid samples: got {len(valid_samples)}, "
                f"need at least {min_samples}"
            )

        neck_reference = median(sample[0] for sample in valid_samples)
        arm_reference = median(sample[1] for sample in valid_samples)
        head_reference = median(sample[2] for sample in valid_samples)

        return cls(
            neck_reference=neck_reference,
            arm_reference=arm_reference,
            head_reference=head_reference,
            sample_count=len(valid_samples),
        )