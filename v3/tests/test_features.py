
import numpy as np
import pytest

from vision.landmarks import PoseLandmarks
from posture.calibration import Calibration
from posture.features import (
    FEATURE_NAMES,
    extract_features,
    validate_feature_sample,
)


def create_valid_pose():
    keypoints = np.zeros((17, 3), dtype=np.float32)

    # Give every landmark valid coordinates and confidence.
    keypoints[:, 0] = 200
    keypoints[:, 1] = 200
    keypoints[:, 2] = 0.9

    # Give selected landmarks different coordinates
    # so that angle calculations are meaningful.
    keypoints[0, :2] = [250, 100]   # Nose
    keypoints[2, :2] = [240, 110]   # Right eye
    keypoints[4, :2] = [230, 120]   # Right ear
    keypoints[6, :2] = [200, 200]   # Right shoulder
    keypoints[8, :2] = [250, 250]   # Right elbow
    keypoints[10, :2] = [300, 250]  # Right wrist

    return PoseLandmarks(keypoints, 640, 480)


def create_valid_calibration():
    return Calibration(
        neck_reference=20.0,
        arm_reference=90.0,
        head_reference=10.0,
        sample_count=10,
    )

def test_extract_features_returns_all_12_features():
    pose = create_valid_pose()
    calibration = create_valid_calibration()

    features = extract_features(pose, calibration)

    assert len(features) == 12
    assert set(features.keys()) == set(FEATURE_NAMES)

def test_extract_features_calculates_angles_and_deviations():
    pose = create_valid_pose()
    calibration = create_valid_calibration()

    features = extract_features(pose, calibration)

    # Verify all three angles were calculated.
    assert features["neck_angle"] is not None
    assert features["arm_angle"] is not None
    assert features["head_angle"] is not None

    # Verify deviations use absolute differences.
    assert features["neck_deviation"] == pytest.approx(
        abs(features["neck_angle"] - calibration.neck_reference)
    )

    assert features["arm_deviation"] == pytest.approx(
        abs(features["arm_angle"] - calibration.arm_reference)
    )

    assert features["head_deviation"] == pytest.approx(
        abs(features["head_angle"] - calibration.head_reference)
    )

def test_extract_features_returns_individual_confidence_scores():
    pose = create_valid_pose()
    calibration = create_valid_calibration()

    # Assign different confidence scores to each selected landmark.
    pose.keypoints[0, 2] = 0.95   # Nose
    pose.keypoints[2, 2] = 0.85   # Right eye
    pose.keypoints[4, 2] = 0.75   # Right ear
    pose.keypoints[6, 2] = 0.65   # Right shoulder
    pose.keypoints[8, 2] = 0.55   # Right elbow
    pose.keypoints[10, 2] = 0.18  # Right wrist

    features = extract_features(pose, calibration)

    assert features["nose_confidence"] == pytest.approx(0.95)
    assert features["right_eye_confidence"] == pytest.approx(0.85)
    assert features["right_ear_confidence"] == pytest.approx(0.75)
    assert features["right_shoulder_confidence"] == pytest.approx(0.65)
    assert features["right_elbow_confidence"] == pytest.approx(0.55)
    assert features["right_wrist_confidence"] == pytest.approx(0.18)

def test_validate_feature_sample_accepts_complete_sample():
    pose = create_valid_pose()
    calibration = create_valid_calibration()

    features = extract_features(pose, calibration)

    assert validate_feature_sample(features) is True

def test_validate_feature_sample_rejects_none_value():
    pose = create_valid_pose()
    calibration = create_valid_calibration()

    features = extract_features(pose, calibration)

    features["neck_angle"] = None

    assert validate_feature_sample(features) is False

def test_validate_feature_sample_rejects_missing_feature():
    features = {name: 1.0 for name in FEATURE_NAMES}
    del features["neck_angle"]

    assert validate_feature_sample(features) is False

def test_validate_feature_sample_rejects_unexpected_feature():
    features = {name: 1.0 for name in FEATURE_NAMES}
    features["extra_feature"] = 1.0

    assert validate_feature_sample(features) is False

def test_validate_feature_sample_rejects_infinite_values():
    features = {name: 1.0 for name in FEATURE_NAMES}
    features["neck_angle"] = float("inf")

    assert validate_feature_sample(features) is False

def test_validate_feature_sample_rejects_nan_values():
    features = {name: 1.0 for name in FEATURE_NAMES}
    features["neck_angle"] = float("nan")

    assert validate_feature_sample(features) is False

def test_validate_feature_sample_rejects_boolean_values():
    features = {name: 1.0 for name in FEATURE_NAMES}
    features["neck_angle"] = True

    assert validate_feature_sample(features) is False
