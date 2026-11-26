
import math
from typing import Optional, Sequence

from vision.landmarks import PoseLandmarks

def calculate_angle(
    v1: Optional[Sequence[float]],
    v2: Optional[Sequence[float]]
) -> Optional[float]:
    """Calculate the angle between two 2D vectors, in degrees."""

    if v1 is None or v2 is None:
        return None

    if len(v1) != 2 or len(v2) != 2:
        return None

    try:
        x1, y1 = float(v1[0]), float(v1[1])
        x2, y2 = float(v2[0]), float(v2[1])
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None

    magnitude1 = math.hypot(x1, y1)
    magnitude2 = math.hypot(x2, y2)

    if magnitude1 == 0 or magnitude2 == 0:
        return None

    dot_product = x1 * x2 + y1 * y2
    cosine = dot_product / (magnitude1 * magnitude2)

    # Protect against tiny floating-point errors.
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def calculate_neck_angle(pose: PoseLandmarks) -> Optional[float]:
    """Calculate the neck angle using the right eye and right shoulder."""

    eye = pose.get("right_eye")
    shoulder = pose.get("right_shoulder")

    if eye is None or shoulder is None:
        return None

    neck_vector = (
        eye.x - shoulder.x,
        eye.y - shoulder.y,
    )

    vertical_vector = (0, -1)

    angle = calculate_angle(neck_vector, vertical_vector)

    if angle is None:
        return None

    # Match V2's acute-angle convention.
    return min(angle, 180 - angle)


def calculate_arm_angle(pose: PoseLandmarks) -> Optional[float]:
    """Calculate the right arm angle using shoulder, elbow, and wrist."""

    shoulder = pose.get("right_shoulder")
    elbow = pose.get("right_elbow")
    wrist = pose.get("right_wrist")

    if shoulder is None or elbow is None or wrist is None:
        return None

    shoulder_vector = (
        shoulder.x - elbow.x,
        shoulder.y - elbow.y,
    )

    arm_vector = (
        elbow.x - wrist.x,
        elbow.y - wrist.y,
    )

    angle = calculate_angle(shoulder_vector, arm_vector)

    if angle is None:
        return None

    # Match V2's obtuse-angle convention.
    return max(angle, 180 - angle)


def calculate_head_angle(pose: PoseLandmarks) -> Optional[float]:
    """Calculate the head angle using the nose and right ear."""

    nose = pose.get("nose")
    ear = pose.get("right_ear")

    if nose is None or ear is None:
        return None

    head_vector = (
        nose.x - ear.x,
        nose.y - ear.y,
    )

    horizontal_vector = (-1, 0)

    angle = calculate_angle(head_vector, horizontal_vector)

    if angle is None:
        return None

    # Match V2's acute-angle convention.
    return min(angle, 180 - angle)
