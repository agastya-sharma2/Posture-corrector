
from dataclasses import dataclass

import numpy as np


@dataclass
class Landmark:
    """Represents one detected body landmark."""

    x: float
    y: float
    confidence: float


class PoseLandmarks:
    """
    Provides named access and validation for MoveNet landmarks.

    MoveNet output format:
        (17, 3)

    Each row contains:
        [x, y, confidence]

    Coordinates are in original camera-image pixels.
    """

    KEYPOINT_INDICES = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    def __init__(
        self,
        keypoints,
        image_width,
        image_height,
        confidence_threshold=0.25
    ):

        self.keypoints = np.asarray(keypoints, dtype=np.float32)

        self.image_width = image_width
        self.image_height = image_height
        self.confidence_threshold = confidence_threshold

        if self.keypoints.shape != (17, 3):
            raise ValueError(
                "Expected keypoints with shape (17, 3), "
                f"got {self.keypoints.shape}"
            )

        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive.")

        if not 0 <= confidence_threshold <= 1:
            raise ValueError(
                "Confidence threshold must be between 0 and 1."
            )

    def get(self, name):
        """
        Return a valid Landmark, or None if it is unreliable.

        A landmark is valid when:
        - Its confidence meets the threshold.
        - Its coordinates are finite.
        - Its coordinates are inside the image.
        """

        if name not in self.KEYPOINT_INDICES:
            raise ValueError(f"Unknown landmark: {name}")

        index = self.KEYPOINT_INDICES[name]

        x, y, confidence = self.keypoints[index]

        # Reject NaN and infinite values.
        if not np.all(np.isfinite([x, y, confidence])):
            return None

        # Reject unreliable detections.
        if confidence < self.confidence_threshold:
            return None

        # Reject invalid confidence values.
        if not 0 <= confidence <= 1:
            return None

        # Reject coordinates outside the camera frame.
        if not (
            0 <= x < self.image_width
            and 0 <= y < self.image_height
        ):
            return None

        return Landmark(
            x=float(x),
            y=float(y),
            confidence=float(confidence)
        )

    def is_valid(self, name):
        """Check whether a named landmark is usable."""

        return self.get(name) is not None

    @property
    def nose(self):
        return self.get("nose")

    @property
    def right_eye(self):
        return self.get("right_eye")

    @property
    def right_ear(self):
        return self.get("right_ear")

    @property
    def right_shoulder(self):
        return self.get("right_shoulder")

    @property
    def right_elbow(self):
        return self.get("right_elbow")

    @property
    def right_wrist(self):
        return self.get("right_wrist")
