
import csv
import time
import uuid
from datetime import datetime
from pathlib import Path

from posture.features import validate_feature_sample


POSTURE_LABELS = {
    "neutral",
    "forward_head",
    "slouched",
}

FEATURE_COLUMNS = [
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
]

CSV_COLUMNS = [
    "sample_id",
    "timestamp",
    "session_id",
    "subject_id",
    "calibration_id",
    "posture_label",
    *FEATURE_COLUMNS,
]


class PostureDataCollector:
    def __init__(
        self,
        csv_path,
        session_id,
        subject_id,
        calibration_id,
        posture_label,
        sample_interval=2.0,
    ):
        if posture_label not in POSTURE_LABELS:
            raise ValueError(f"Invalid posture label: {posture_label}")

        if sample_interval <= 0:
            raise ValueError("Sample interval must be positive.")

        self.csv_path = Path(csv_path)
        self.session_id = session_id
        self.subject_id = subject_id
        self.calibration_id = calibration_id
        self.posture_label = posture_label
        self.sample_interval = sample_interval

        self.last_sample_time = None

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_csv()

    def _initialize_csv(self):
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with self.csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(CSV_COLUMNS)
            return

        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            existing_columns = next(reader, [])

        if existing_columns != CSV_COLUMNS:
            raise ValueError("Existing CSV has an unexpected column structure.")

    def add_sample(self, features):
        """
        Save a valid feature sample if the sampling interval has elapsed.

        Returns True if a row was saved, otherwise False.
        """

        if not isinstance(features, dict):
            return False

        if not validate_feature_sample(features):
            return False

        current_time = time.monotonic()

        if self.last_sample_time is not None:
            elapsed = current_time - self.last_sample_time

            if elapsed < self.sample_interval:
                return False

        row = {
            "sample_id": uuid.uuid4().hex,
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "calibration_id": self.calibration_id,
            "posture_label": self.posture_label,
        }

        row.update(features)

        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

        self.last_sample_time = current_time
        return True
