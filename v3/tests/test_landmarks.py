
import numpy as np
import pytest

from vision.landmarks import PoseLandmarks


def create_keypoints():
    """Create 17 valid dummy landmarks."""

    points = np.zeros((17, 3), dtype=np.float32)

    points[:, 0] = 200  # x
    points[:, 1] = 150  # y
    points[:, 2] = 0.90  # confidence

    return points

def create_valid_keypoints():
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 0] = 100
    keypoints[:, 1] = 100
    keypoints[:, 2] = 0.9
    return keypoints

def test_valid_landmark():
    points = create_keypoints()

    pose = PoseLandmarks(
        points,
        image_width=640,
        image_height=480
    )

    assert pose.is_valid("right_shoulder")

    shoulder = pose.right_shoulder

    assert shoulder is not None
    assert shoulder.x == 200
    assert shoulder.y == 150
    assert shoulder.confidence == pytest.approx(0.90)


def test_low_confidence_landmark():
    points = create_keypoints()

    points[10, 2] = 0.12  # right wrist

    pose = PoseLandmarks(points, 640, 480)

    assert not pose.is_valid("right_wrist")
    assert pose.right_wrist is None


def test_landmark_outside_image():
    points = create_keypoints()

    points[6, 0] = 700  # right shoulder x-coordinate

    pose = PoseLandmarks(points, 640, 480)

    assert not pose.is_valid("right_shoulder")


def test_invalid_array_shape():
    points = np.zeros((10, 3))

    with pytest.raises(ValueError):
        PoseLandmarks(points, 640, 480)


def test_unknown_landmark():
    points = create_keypoints()

    pose = PoseLandmarks(points, 640, 480)

    with pytest.raises(ValueError):
        pose.get("unknown_landmark")


def test_nan_coordinate():
    points = create_keypoints()

    points[2, 0] = np.nan  # right eye

    pose = PoseLandmarks(points, 640, 480)

    assert not pose.is_valid("right_eye")


def test_get_confidence_returns_original_score():
    keypoints = create_valid_keypoints()
    keypoints[10, 2] = 0.85

    pose = PoseLandmarks(keypoints, 640, 480)

    assert pose.get_confidence("right_wrist") == pytest.approx(0.85)


def test_get_confidence_returns_low_confidence_score():
    keypoints = create_valid_keypoints()
    keypoints[10, 2] = 0.18

    pose = PoseLandmarks(keypoints, 640, 480)

    assert pose.get("right_wrist") is None
    assert pose.get_confidence("right_wrist") == pytest.approx(0.18)


def test_get_confidence_rejects_invalid_confidence():
    keypoints = create_valid_keypoints()
    keypoints[10, 2] = np.nan

    pose = PoseLandmarks(keypoints, 640, 480)

    assert pose.get_confidence("right_wrist") is None


def test_get_confidence_rejects_unknown_landmark():
    keypoints = create_valid_keypoints()

    pose = PoseLandmarks(keypoints, 640, 480)

    with pytest.raises(ValueError, match="Unknown landmark"):
        pose.get_confidence("unknown")
