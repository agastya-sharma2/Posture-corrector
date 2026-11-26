
import pytest

import numpy as np

from vision.landmarks import PoseLandmarks
from posture.angles import calculate_angle, calculate_neck_angle, calculate_arm_angle, calculate_head_angle


def test_perpendicular_vectors():
    assert calculate_angle((1, 0), (0, 1)) == pytest.approx(90.0)


def test_parallel_vectors():
    assert calculate_angle((1, 0), (1, 0)) == pytest.approx(0.0)


def test_opposite_vectors():
    assert calculate_angle((1, 0), (-1, 0)) == pytest.approx(180.0)


def test_zero_length_vector():
    assert calculate_angle((0, 0), (1, 0)) is None


def test_missing_vector():
    assert calculate_angle(None, (1, 0)) is None


def test_non_finite_vector():
    assert calculate_angle((float("nan"), 0), (1, 0)) is None

def make_pose(eye, shoulder):
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 2] = 1.0

    # MoveNet indices:
    # Right eye = 2
    # Right shoulder = 6
    keypoints[2, :2] = eye
    keypoints[6, :2] = shoulder

    return PoseLandmarks(
        keypoints,
        image_width=640,
        image_height=480
    )


def test_neck_angle_when_vertical():
    pose = make_pose(
        eye=(100, 100),
        shoulder=(100, 200)
    )

    assert calculate_neck_angle(pose) == pytest.approx(0.0)


def test_neck_angle_when_tilted():
    pose = make_pose(
        eye=(150, 100),
        shoulder=(100, 200)
    )

    assert calculate_neck_angle(pose) == pytest.approx(26.565, abs=0.001)


def make_arm_pose(shoulder, elbow, wrist):
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 2] = 1.0

    # MoveNet indices:
    # Right shoulder = 6
    # Right elbow = 8
    # Right wrist = 10
    keypoints[6, :2] = shoulder
    keypoints[8, :2] = elbow
    keypoints[10, :2] = wrist

    return PoseLandmarks(
        keypoints,
        image_width=640,
        image_height=480
    )


def test_arm_angle_when_straight():
    pose = make_arm_pose(
        shoulder=(100, 100),
        elbow=(150, 100),
        wrist=(200, 100)
    )

    assert calculate_arm_angle(pose) == pytest.approx(180.0)


def test_arm_angle_when_bent_90_degrees():
    pose = make_arm_pose(
        shoulder=(100, 100),
        elbow=(150, 100),
        wrist=(150, 150)
    )

    assert calculate_arm_angle(pose) == pytest.approx(90.0)


def test_arm_angle_with_zero_length_vector():
    pose = make_arm_pose(
        shoulder=(150, 100),
        elbow=(150, 100),
        wrist=(200, 100)
    )

    assert calculate_arm_angle(pose) is None


def make_head_pose(nose, ear):
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 2] = 1.0

    # MoveNet indices:
    # Nose = 0
    # Right ear = 4
    keypoints[0, :2] = nose
    keypoints[4, :2] = ear

    return PoseLandmarks(
        keypoints,
        image_width=640,
        image_height=480
    )


def test_head_angle_when_horizontal():
    pose = make_head_pose(
        nose=(100, 100),
        ear=(200, 100)
    )

    assert calculate_head_angle(pose) == pytest.approx(0.0)


def test_head_angle_when_tilted():
    pose = make_head_pose(
        nose=(100, 50),
        ear=(200, 100)
    )

    assert calculate_head_angle(pose) == pytest.approx(26.565, abs=0.001)


def test_head_angle_with_missing_nose():
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 2] = 1.0

    keypoints[0, 2] = 0.1  # Below the 0.25 confidence threshold.
    keypoints[4, :2] = (200, 100)

    pose = PoseLandmarks(keypoints, 640, 480)

    assert calculate_head_angle(pose) is None
