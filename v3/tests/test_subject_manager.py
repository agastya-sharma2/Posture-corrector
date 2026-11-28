
import pytest

from posture.subject_manager import SubjectManager
from posture.calibration import Calibration


@pytest.fixture
def manager(tmp_path):
    return SubjectManager(
        profiles_directory=tmp_path / "profiles"
    )


@pytest.mark.parametrize(
    "subject_id",
    [
        "subject_001",
        "subject-001",
        "participant1",
        "ABC123",
    ],
)
def test_valid_subject_ids(manager, subject_id):
    assert manager.validate_subject_id(subject_id) is True


@pytest.mark.parametrize(
    "subject_id",
    [
        "",
        "../subject_001",
        "subject/001",
        "subject 001",
        "subject.001",
        None,
        123,
    ],
)
def test_invalid_subject_ids(manager, subject_id):
    assert manager.validate_subject_id(subject_id) is False


def test_get_profile_path(manager):
    path = manager.get_profile_path("subject_001")

    assert path == manager.profiles_directory / "subject_001.json"


def test_get_profile_path_rejects_invalid_id(manager):
    with pytest.raises(ValueError):
        manager.get_profile_path("../outside")


def test_profile_does_not_exist(manager):
    assert manager.profile_exists("subject_001") is False


def test_profile_exists(manager):
    manager.profiles_directory.mkdir(parents=True)
    profile = manager.profiles_directory / "subject_001.json"
    profile.write_text("{}", encoding="utf-8")

    assert manager.profile_exists("subject_001") is True


def test_save_and_load_subject_calibration(manager):
    calibration = Calibration(
        neck_reference=15.0,
        arm_reference=90.0,
        head_reference=10.0,
        sample_count=10,
    )

    manager.save_calibration("subject_001", calibration)

    loaded_calibration, settings = manager.load_calibration(
        "subject_001"
    )

    assert loaded_calibration == calibration
    assert settings["confidence_threshold"] == 0.25


def test_load_calibration_rejects_missing_profile(manager):
    with pytest.raises(FileNotFoundError):
        manager.load_calibration("subject_001")


def test_subjects_have_separate_calibration_profiles(manager):
    calibration_1 = Calibration(15.0, 90.0, 10.0, 10)
    calibration_2 = Calibration(20.0, 85.0, 12.0, 10)

    manager.save_calibration("subject_001", calibration_1)
    manager.save_calibration("subject_002", calibration_2)

    loaded_1, _ = manager.load_calibration("subject_001")
    loaded_2, _ = manager.load_calibration("subject_002")

    assert loaded_1 == calibration_1
    assert loaded_2 == calibration_2
    assert loaded_1 != loaded_2
