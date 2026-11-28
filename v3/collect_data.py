import time
import uuid
from pathlib import Path

from camera.camera_manager import CameraManager
from vision.movenet import MoveNet
from vision.landmarks import PoseLandmarks

from posture.angles import (
    calculate_neck_angle,
    calculate_arm_angle,
    calculate_head_angle,
)
from posture.analyzer import analyze_posture
from posture.calibration_collector import CalibrationCollector
from posture.collection_session import (
    CollectionSession,
    POSTURE_LABELS,
)
from posture.data_collector import PostureDataCollector
from posture.subject_manager import SubjectManager


MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "movenet_thunder.tflite"
)

CONFIDENCE_THRESHOLD = 0.25

POSTURE_OPTIONS = {
    "1": "neutral",
    "2": "forward_head",
    "3": "slouched",
}

POSTURE_INSTRUCTIONS = {
    "neutral": (
        "Sit in your normal, comfortable working posture."
    ),
    "forward_head": (
        "Demonstrate a forward-head posture while seated. "
        "Keep the position comfortable."
    ),
    "slouched": (
        "Demonstrate a slouched sitting posture. "
        "Do not force or exaggerate the position."
    ),
}

def get_subject_id():
    manager = SubjectManager()

    while True:
        subject_id = input("\nEnter subject ID: ").strip()

        if manager.validate_subject_id(subject_id):
            return subject_id

        print(
            "Invalid subject ID. Use only letters, "
            "numbers, underscores, and hyphens."
        )


def get_posture_label():
    print("\nSelect posture to record:")
    print("1. Neutral")
    print("2. Forward head")
    print("3. Slouched")

    while True:
        choice = input("\nSelection: ").strip()

        if choice in POSTURE_OPTIONS:
            return POSTURE_OPTIONS[choice]

        print("Invalid selection. Please choose 1, 2, or 3.")


def calibrate_subject(pose_detector, camera):
    collector = CalibrationCollector(
        target_samples=10,
        min_samples=5,
    )

    print("\nCalibration starting.")
    print("Sit in your normal, comfortable working posture.")
    print("Collecting 10 valid samples...\n")

    while not collector.is_complete():
        frame = camera.capture_frame()
        landmarks = pose_detector.detect(frame)

        pose = PoseLandmarks(
            keypoints=landmarks,
            image_width=frame.shape[1],
            image_height=frame.shape[0],
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

        angles = [
            calculate_neck_angle(pose),
            calculate_arm_angle(pose),
            calculate_head_angle(pose),
        ]

        if collector.add_sample(angles):
            print(f"Accepted sample: {len(collector.samples)}/10")
        else:
            print("Invalid sample. Trying again.")

    return collector.get_calibration()


def run_collection_session(
    camera,
    pose_detector,
    calibration,
    subject_id,
    posture_label,
    duration=120,
    countdown=5,
    csv_path=None,
):
    session_id = f"{subject_id}_{uuid.uuid4().hex[:8]}"
    calibration_id = f"{subject_id}_v1"

    if csv_path is None:
        csv_path = (
            Path(__file__).parent
            / "data"
            / "datasets"
            / "posture_dataset_v1.csv"
        )

    collector = PostureDataCollector(
        csv_path=csv_path,
        session_id=session_id,
        subject_id=subject_id,
        calibration_id=calibration_id,
        posture_label=posture_label,
        sample_interval=2.0,
    )

    session = CollectionSession(
        subject_id=subject_id,
        posture_label=posture_label,
        duration=duration,
        countdown=countdown,
    )

    print("\nGet ready!")
    print(f"Recording posture: {posture_label}")
    print(POSTURE_INSTRUCTIONS[posture_label])

    for remaining in range(countdown, 0, -1):
        print(f"Starting in {remaining}...")
        time.sleep(1)

    print("\nRecording started. Press Ctrl+C to stop.\n")

    session.start()

    samples_saved = 0
    invalid_frames = 0
    next_update = 0

    try:
        while not session.is_complete():
            frame = camera.capture_frame()
            landmarks = pose_detector.detect(frame)

            pose = PoseLandmarks(
                keypoints=landmarks,
                image_width=frame.shape[1],
                image_height=frame.shape[0],
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )

            result = analyze_posture(pose, calibration)

            if result.is_valid:
                if collector.add_sample(result.features):
                    samples_saved += 1
            else:
                invalid_frames += 1

            elapsed = session.elapsed_time()

            if elapsed >= next_update:
                remaining = int(session.remaining_time())

                print(
                    f"Time remaining: {remaining:3d}s | "
                    f"Samples saved: {samples_saved:3d} | "
                    f"Invalid frames: {invalid_frames:3d}",
                    end="\r",
                    flush=True,
                )

                next_update += 1

    except KeyboardInterrupt:
        session.finish()
        print("\n\nRecording interrupted.")
        raise

    print("\n\nSession complete.")
    print(f"Posture: {posture_label}")
    print(f"Samples saved: {samples_saved}")
    print(f"Invalid frames: {invalid_frames}")

    return samples_saved


def main():
    print("=" * 40)
    print("       POSTURE DATA COLLECTION")
    print("=" * 40)

    subject_manager = SubjectManager()
    subject_id = get_subject_id()

    print(f"\nSubject: {subject_id}")

    pose_detector = MoveNet(
        model_path=str(MODEL_PATH),
        num_threads=2,
    )

    try:
        with CameraManager() as camera:

            if subject_manager.profile_exists(subject_id):
                print("Existing calibration found.")
                print("Loading subject calibration...")

                calibration, settings = subject_manager.load_calibration(
                    subject_id
                )

            else:
                print("No calibration profile found.")
                print("Starting new calibration...")

                calibration = calibrate_subject(
                    pose_detector,
                    camera,
                )

                subject_manager.save_calibration(
                    subject_id,
                    calibration,
                )

                settings = {}

                print("\nCalibration saved successfully.")

            print("\nCalibration references:")
            print(f"Neck: {calibration.neck_reference:.1f}")
            print(f"Arm:  {calibration.arm_reference:.1f}")
            print(f"Head: {calibration.head_reference:.1f}")

            print("\nSubject setup complete.")

            total_samples = 0

            for posture_number in range(3):
                print(f"\n{'=' * 40}")
                print(f"POSTURE {posture_number + 1}/3")
                print(f"{'=' * 40}")

                posture_label = get_posture_label()

                print(f"\nSelected posture: {posture_label}")

                total_samples += run_collection_session(
                    camera=camera,
                    pose_detector=pose_detector,
                    calibration=calibration,
                    subject_id=subject_id,
                    posture_label=posture_label,
                )

            print("\n" + "=" * 40)
            print("SUBJECT COLLECTION COMPLETE")
            print("=" * 40)
            print(f"Subject: {subject_id}")
            print(f"Total samples saved: {total_samples}")

    except KeyboardInterrupt:
        print("\nSubject collection interrupted.")
        print("Previously saved CSV samples are preserved.")

if __name__ == "__main__":
    main()
