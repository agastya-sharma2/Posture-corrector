
import re
from pathlib import Path

from posture.profile_manager import save_profile, load_profile


SUBJECT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class SubjectManager:
    def __init__(self, profiles_directory="data/profiles"):
        self.profiles_directory = Path(profiles_directory)

    @staticmethod
    def validate_subject_id(subject_id):
        if not isinstance(subject_id, str):
            return False

        if not subject_id:
            return False

        return bool(SUBJECT_ID_PATTERN.fullmatch(subject_id))

    def get_profile_path(self, subject_id):
        if not self.validate_subject_id(subject_id):
            raise ValueError("Invalid subject ID.")

        return self.profiles_directory / f"{subject_id}.json"

    def profile_exists(self, subject_id):
        profile_path = self.get_profile_path(subject_id)
        return profile_path.is_file()

    def load_calibration(self, subject_id):
        profile_path = self.get_profile_path(subject_id)

        if not profile_path.is_file():
            raise FileNotFoundError(
                f"No calibration profile found for {subject_id}."
            )

        calibration, settings = load_profile(profile_path)

        return calibration, settings

    def save_calibration(self, subject_id, calibration):
        profile_path = self.get_profile_path(subject_id)

        save_profile(profile_path, calibration)
