
import math

import pytest

from posture.calibration import Calibration


def test_calibration_uses_median_of_each_angle():
    samples = [
        (12.0, 85.0, 8.0),
        (13.0, 86.0, 9.0),
        (12.5, 84.0, 8.5),
        (50.0, 140.0, 60.0),
        (12.0, 85.0, 8.0),
    ]

    calibration = Calibration.from_samples(samples)

    assert calibration.neck_reference == 12.5
    assert calibration.arm_reference == 85.0
    assert calibration.head_reference == 8.5
    assert calibration.sample_count == 5


def test_calibration_ignores_samples_with_missing_angles():
    samples = [
        (12.0, 85.0, 8.0),
        (None, 86.0, 9.0),
        (13.0, 87.0, 10.0),
        (12.0, None, 8.0),
        (14.0, 88.0, 11.0),
    ]

    calibration = Calibration.from_samples(samples, min_samples=3)

    assert calibration.sample_count == 3


def test_calibration_ignores_invalid_angle_values():
    samples = [
        (12.0, 85.0, 8.0),
        (math.nan, 86.0, 9.0),
        (13.0, math.inf, 10.0),
        (-1.0, 87.0, 11.0),
        (14.0, 88.0, 181.0),
    ]

    calibration = Calibration.from_samples(samples, min_samples=1)

    assert calibration.sample_count == 1


def test_calibration_rejects_samples_with_wrong_length():
    samples = [
        (12.0, 85.0, 8.0),
        (13.0, 86.0),
        (14.0, 87.0, 9.0, 10.0),
    ]

    calibration = Calibration.from_samples(samples, min_samples=1)

    assert calibration.sample_count == 1


def test_calibration_rejects_insufficient_valid_samples():
    samples = [
        (12.0, 85.0, 8.0),
        (None, 86.0, 9.0),
    ]

    with pytest.raises(ValueError, match="Not enough valid samples"):
        Calibration.from_samples(samples, min_samples=2)


def test_calibration_rejects_invalid_minimum_sample_count():
    with pytest.raises(ValueError, match="min_samples"):
        Calibration.from_samples([(12.0, 85.0, 8.0)], min_samples=0)


def test_calibration_records_valid_sample_count():
    samples = [
        (12.0, 85.0, 8.0),
        (13.0, 86.0, 9.0),
        (None, 87.0, 10.0),
        (14.0, 88.0, 11.0),
    ]

    calibration = Calibration.from_samples(samples, min_samples=3)

    assert calibration.sample_count == 3
