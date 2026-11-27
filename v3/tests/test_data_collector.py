
import csv
import time

import pytest

from posture.data_collector import (
    PostureDataCollector,
    CSV_COLUMNS,
    FEATURE_COLUMNS,
)


@pytest.fixture
def valid_features():
    return {
        "neck_angle": 15.0,
        "arm_angle": 90.0,
        "head_angle": 10.0,
        "nose_confidence": 0.9,
        "right_eye_confidence": 0.9,
        "right_ear_confidence": 0.9,
        "right_shoulder_confidence": 0.9,
        "right_elbow_confidence": 0.9,
        "right_wrist_confidence": 0.9,
        "neck_deviation": 5.0,
        "arm_deviation": 10.0,
        "head_deviation": 3.0,
    }


@pytest.fixture
def collector(tmp_path):
    return PostureDataCollector(
        csv_path=tmp_path / "dataset.csv",
        session_id="session_001",
        subject_id="subject_001",
        calibration_id="subject_001_v1",
        posture_label="neutral",
        sample_interval=2.0,
    )


def test_collector_creates_csv_with_correct_headers(collector):
    with collector.csv_path.open() as file:
        reader = csv.reader(file)
        headers = next(reader)

    assert headers == CSV_COLUMNS


def test_collector_saves_valid_sample(collector, valid_features):
    assert collector.add_sample(valid_features) is True

    with collector.csv_path.open() as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["posture_label"] == "neutral"
    assert rows[0]["subject_id"] == "subject_001"


def test_collector_rejects_invalid_sample(collector):
    assert collector.add_sample(None) is False

    with collector.csv_path.open() as file:
        rows = list(csv.DictReader(file))

    assert rows == []


def test_collector_enforces_sampling_interval(
    collector, valid_features
):
    assert collector.add_sample(valid_features) is True

    assert collector.add_sample(valid_features) is False

    time.sleep(2.1)

    assert collector.add_sample(valid_features) is True


def test_collector_saves_all_feature_columns(
    collector, valid_features
):
    collector.add_sample(valid_features)

    with collector.csv_path.open() as file:
        row = next(csv.DictReader(file))

    for feature in FEATURE_COLUMNS:
        assert float(row[feature]) == valid_features[feature]


def test_collector_appends_samples_without_overwriting(
    collector, valid_features
):
    collector.add_sample(valid_features)

    collector.last_sample_time -= 2.1
    assert collector.add_sample(valid_features) is True

    with collector.csv_path.open() as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2
    assert rows[0]["sample_id"] != rows[1]["sample_id"]


def test_collector_rejects_invalid_posture_label(tmp_path):
    with pytest.raises(ValueError):
        PostureDataCollector(
            csv_path=tmp_path / "dataset.csv",
            session_id="session_001",
            subject_id="subject_001",
            calibration_id="subject_001_v1",
            posture_label="unknown",
        )


def test_invalid_sample_does_not_reset_sampling_timer(
    collector, valid_features
):
    assert collector.add_sample(valid_features) is True

    original_sample_time = collector.last_sample_time

    assert collector.add_sample(None) is False

    assert collector.last_sample_time == original_sample_time


def test_invalid_features_are_not_saved(
    collector, valid_features
):
    invalid_features = valid_features.copy()
    invalid_features["neck_angle"] = None

    assert collector.add_sample(invalid_features) is False

    with collector.csv_path.open() as file:
        rows = list(csv.DictReader(file))

    assert rows == []
