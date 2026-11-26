
from dataclasses import dataclass
from typing import Optional

from posture.calibration import Calibration
from posture.features import extract_features, validate_feature_sample
from vision.landmarks import PoseLandmarks


@dataclass(frozen=True)
class AnalysisResult:
    features: Optional[dict]
    is_valid: bool
    reason: Optional[str]


def analyze_posture(
    pose: PoseLandmarks,
    calibration: Calibration,
) -> AnalysisResult:
    """
    Analyze a pose using the user's personal calibration.

    Extracts posture features and validates the resulting sample.

    Returns an AnalysisResult containing either a valid feature
    dictionary or a rejection reason.
    """

    features = extract_features(pose, calibration)

    if not validate_feature_sample(features):
        return AnalysisResult(
            features=None,
            is_valid=False,
            reason="Missing or invalid feature values",
        )

    return AnalysisResult(
        features=features,
        is_valid=True,
        reason=None,
    )
