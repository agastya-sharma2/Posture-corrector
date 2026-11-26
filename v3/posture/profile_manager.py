import json
from pathlib import Path

from posture.calibration import Calibration

def save_profile(path: str | Path, calibration: Calibration):
    profile = {
        "profile_version": 1,
        "calibration": {
            "neck_reference": calibration.neck_reference,
            "arm_reference": calibration.arm_reference,
            "head_reference": calibration.head_reference,
            "sample_count": calibration.sample_count,
        },
        "settings": {
            "confidence_threshold": 0.25
        }
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=4)

def load_profile(path: str | Path) -> tuple[Calibration, dict]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        profile = json.load(file)

    if profile.get("profile_version") != 1:
        raise ValueError("Unsupported profile version")

    calibration_data = profile["calibration"]

    calibration = Calibration(
        neck_reference=calibration_data["neck_reference"],
        arm_reference=calibration_data["arm_reference"],
        head_reference=calibration_data["head_reference"],
        sample_count=calibration_data["sample_count"],
    )

    settings = profile.get("settings", {})

    return calibration, settings