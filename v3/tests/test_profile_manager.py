import json
import pytest

from posture.calibration import Calibration
from posture.profile_manager import save_profile, load_profile

def test_save_profile_creates_json_file(tmp_path):
    calibration = Calibration(20.5, 92.3, 11.2, 10)
    path = tmp_path / "default.json"

    save_profile(path, calibration)

    assert path.exists()

    with path.open() as file:
        profile = json.load(file)

    assert profile["profile_version"] == 1
    assert profile["calibration"]["neck_reference"] == 20.5
    assert profile["calibration"]["sample_count"] == 10


def test_load_profile_returns_calibration_and_settings(tmp_path):
    calibration = Calibration(20.5, 92.3, 11.2, 10)
    path = tmp_path / "default.json"

    save_profile(path, calibration)

    loaded_calibration, settings = load_profile(path)

    assert loaded_calibration == calibration
    assert settings["confidence_threshold"] == 0.25


def test_load_profile_rejects_unsupported_version(tmp_path):
    path = tmp_path / "default.json"
    path.write_text(
        '{"profile_version": 2}',
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unsupported profile version"):
        load_profile(path)


def test_load_profile_rejects_missing_calibration(tmp_path):
    path = tmp_path / "default.json"
    path.write_text(
        '{"profile_version": 1}',
        encoding="utf-8"
    )

    with pytest.raises(KeyError):
        load_profile(path)