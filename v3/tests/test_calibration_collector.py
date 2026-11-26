import pytest

from posture.calibration_collector import CalibrationCollector
from posture.calibration import Calibration

def test_add_valid_sample():
    collector = CalibrationCollector()

    result = collector.add_sample([20.0, 90.0, 10.0])

    assert result is True
    assert len(collector.samples) == 1


def test_rejects_missing_sample():
    collector = CalibrationCollector()

    assert collector.add_sample(None) is False
    assert len(collector.samples) == 0


def test_rejects_wrong_number_of_angles():
    collector = CalibrationCollector()

    assert collector.add_sample([20.0, 90.0]) is False
    assert len(collector.samples) == 0


def test_rejects_invalid_angle():
    collector = CalibrationCollector()

    assert collector.add_sample([20.0, None, 10.0]) is False
    assert len(collector.samples) == 0


def test_rejects_angle_outside_valid_range():
    collector = CalibrationCollector()

    assert collector.add_sample([20.0, 190.0, 10.0]) is False
    assert len(collector.samples) == 0


def test_collector_is_not_complete_initially():
    collector = CalibrationCollector()

    assert collector.is_complete() is False


def test_collector_stops_accepting_samples_at_target():
    collector = CalibrationCollector(target_samples=2, min_samples=1)

    assert collector.add_sample([20.0, 90.0, 10.0]) is True
    assert collector.add_sample([21.0, 91.0, 11.0]) is True

    assert collector.is_complete() is True

    assert collector.add_sample([22.0, 92.0, 12.0]) is False
    assert len(collector.samples) == 2


def test_get_calibration_rejects_insufficient_samples():
    collector = CalibrationCollector(
        target_samples=10,
        min_samples=5
    )

    for _ in range(4):
        collector.add_sample([20.0, 90.0, 10.0])

    with pytest.raises(ValueError, match="Not enough valid samples"):
        collector.get_calibration()


def test_get_calibration_returns_correct_references():
    collector = CalibrationCollector(
        target_samples=10,
        min_samples=5
    )

    samples = [
        [20.0, 90.0, 10.0],
        [21.0, 92.0, 11.0],
        [19.0, 88.0, 9.0],
        [22.0, 91.0, 12.0],
        [20.0, 89.0, 10.0],
    ]

    for sample in samples:
        assert collector.add_sample(sample) is True

    calibration = collector.get_calibration()

    assert isinstance(calibration, Calibration)
    assert calibration.neck_reference == 20.0
    assert calibration.arm_reference == 90.0
    assert calibration.head_reference == 10.0
    assert calibration.sample_count == 5


def test_reset_clears_collected_samples():
    collector = CalibrationCollector()

    collector.add_sample([20.0, 90.0, 10.0])
    collector.add_sample([21.0, 91.0, 11.0])

    collector.reset()

    assert len(collector.samples) == 0
    assert collector.is_complete() is False


def test_rejects_zero_minimum_samples():
    with pytest.raises(ValueError, match="min_samples"):
        CalibrationCollector(min_samples=0)


def test_rejects_target_below_minimum():
    with pytest.raises(ValueError, match="target_samples"):
        CalibrationCollector(
            target_samples=4,
            min_samples=5
        )


def test_rejects_non_sequence_input():
    collector = CalibrationCollector()

    assert collector.add_sample(90) is False
    assert len(collector.samples) == 0


def test_rejects_boolean_angles():
    collector = CalibrationCollector()

    assert collector.add_sample([20.0, True, 10.0]) is False
    assert len(collector.samples) == 0