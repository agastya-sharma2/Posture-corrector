
import numpy as np
import pytest

from vision.landmarks import PoseLandmarks
from posture.calibration import Calibration
from posture.analyzer import analyze_posture, AnalysisResult


def create_valid_pose():
    keypoints = np.zeros((17, 3), dtype=np.float32)

    keypoints[:, 0] = 200
    keypoints[:, 1] = 200
    keypoints[:, 2] = 0.9

    # Nose
    keypoints[0, :2] = [250, 100]

    # Right eye
    keypoints[2, :2] = [240, 110]

    # Right ear
    keypoints[4, :2] = [230, 120]

    # Right shoulder
    keypoints[6, :2] = [200, 200]

    # Right elbow
    keypoints[8, :2] = [250, 250]

    # Right wrist
    keypoints[10, :2] = [300, 250]

    return PoseLandmarks(keypoints, 640, 480)


def create_valid_calibration():
    return Calibration(
        neck_reference=20.0,
        arm_reference=90.0,
        head_reference=10.0,
        sample_count=10,
    )

def test_analyze_posture_returns_valid_result():
    pose = create_valid_pose()
    calibration = create_valid_calibration()

    result = analyze_posture(pose, calibration)

    assert isinstance(result, AnalysisResult)
    assert result.is_valid is True
    assert result.features is not None
    assert result.reason is None

def test_analyze_posture_returns_all_12_features():
    pose = create_valid_pose()
    calibration = create_valid_calibration()

    result = analyze_posture(pose, calibration)

    assert len(result.features) == 12
    assert set(result.features.keys()) == {
        "neck_angle",
        "arm_angle",
        "head_angle",
        "nose_confidence",
        "right_eye_confidence",
        "right_ear_confidence",
        "right_shoulder_confidence",
        "right_elbow_confidence",
        "right_wrist_confidence",
        "neck_deviation",
        "arm_deviation",
        "head_deviation",
    }

def test_analyze_posture_rejects_invalid_pose():
    pose = create_valid_pose()
    calibration = create_valid_calibration()

    # Make the right wrist confidence too low.
    pose.keypoints[10, 2] = 0.10

    result = analyze_posture(pose, calibration)

    assert result.is_valid is False
    assert result.features is None
    assert result.reason is not None

def test_analyze_posture_returns_rejection_reason():
    pose = create_valid_pose()
    calibration = create_valid_calibration()

    pose.keypoints[10, 2] = 0.10

    result = analyze_posture(pose, calibration)

    assert result.reason == "Missing or invalid feature values"

def test_analyze_posture_result_fields_are_consistent():
    pose = create_valid_pose()
    calibration = create_valid_calibration()

    result = analyze_posture(pose, calibration)

    if result.is_valid:
        assert result.features is not None
        assert result.reason is None
    else:
        assert result.features is None
        assert result.reason is not None
