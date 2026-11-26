
import math
from typing import Mapping, Optional

from posture.angles import (
    calculate_neck_angle,
    calculate_arm_angle,
    calculate_head_angle,
)
from posture.calibration import Calibration
from vision.landmarks import PoseLandmarks


# Fixed feature order for future ML training.
FEATURE_NAMES = (
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
)


def extract_features(
    pose: PoseLandmarks,
    calibration: Calibration,
) -> dict[str, Optional[float]]:
    """
    Extract the 12 agreed features from a pose and calibration.

    Missing or invalid measurements are represented by None.
    """

    # Calculate current posture angles.
    neck_angle = calculate_neck_angle(pose)
    arm_angle = calculate_arm_angle(pose)
    head_angle = calculate_head_angle(pose)

    # Retrieve individual landmark confidence scores.
    nose_confidence = pose.get_confidence("nose")
    right_eye_confidence = pose.get_confidence("right_eye")
    right_ear_confidence = pose.get_confidence("right_ear")
    right_shoulder_confidence = pose.get_confidence("right_shoulder")
    right_elbow_confidence = pose.get_confidence("right_elbow")
    right_wrist_confidence = pose.get_confidence("right_wrist")

    # Calculate absolute deviations from personal calibration.
    neck_deviation = (
        abs(neck_angle - calibration.neck_reference)
        if neck_angle is not None else None
    )

    arm_deviation = (
        abs(arm_angle - calibration.arm_reference)
        if arm_angle is not None else None
    )

    head_deviation = (
        abs(head_angle - calibration.head_reference)
        if head_angle is not None else None
    )

    return {
        "neck_angle": neck_angle,
        "arm_angle": arm_angle,
        "head_angle": head_angle,
        "nose_confidence": nose_confidence,
        "right_eye_confidence": right_eye_confidence,
        "right_ear_confidence": right_ear_confidence,
        "right_shoulder_confidence": right_shoulder_confidence,
        "right_elbow_confidence": right_elbow_confidence,
        "right_wrist_confidence": right_wrist_confidence,
        "neck_deviation": neck_deviation,
        "arm_deviation": arm_deviation,
        "head_deviation": head_deviation,
    }


def validate_feature_sample(
    features: Mapping[str, Optional[float]],
) -> bool:
    """
    Return True only if all 12 expected features are present
    and contain valid, finite numeric values.
    """

    # Reject missing or unexpected feature names.
    if set(features.keys()) != set(FEATURE_NAMES):
        return False

    # Reject None, non-numeric values, and non-finite numbers.
    for value in features.values():
        if value is None or isinstance(value, bool):
            return False

        try:
            if not math.isfinite(value):
                return False
        except (TypeError, ValueError, OverflowError):
            return False

    return True
