
import json
from pathlib import Path

from camera.camera_manager import CameraManager
from vision.movenet import MoveNet
from vision.landmarks import PoseLandmarks

from posture.angles import (
    calculate_neck_angle,
    calculate_arm_angle,
    calculate_head_angle,
)
from posture.calibration_collector import CalibrationCollector
from posture.analyzer import analyze_posture
from posture.profile_manager import save_profile, load_profile


MODEL_PATH = Path(__file__).parent / "models" / "movenet_thunder.tflite"
PROFILE_PATH = Path(__file__).parent / "data" / "profiles" / "default.json"

CONFIDENCE_THRESHOLD = 0.25


def calibrate(pose_detector, camera):
    collector = CalibrationCollector(
        target_samples=10,
        min_samples=5
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
            confidence_threshold=CONFIDENCE_THRESHOLD
        )

        angles = [
            calculate_neck_angle(pose),
            calculate_arm_angle(pose),
            calculate_head_angle(pose)
        ]

        if collector.add_sample(angles):
            print(f"Accepted sample: {len(collector.samples)}/10")
        else:
            print("Invalid sample. Trying again.")

    return collector.get_calibration()


def load_or_create_calibration(pose_detector, camera):
    if PROFILE_PATH.exists() and PROFILE_PATH.stat().st_size > 0:
        try:
            calibration, _settings = load_profile(PROFILE_PATH)
            print("Existing calibration loaded.")
            return calibration

        except (ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"Could not load profile: {error}")

    calibration = calibrate(pose_detector, camera)
    save_profile(PROFILE_PATH, calibration)

    print("Calibration saved.")
    return calibration


def main():
    pose_detector = MoveNet(
        model_path=str(MODEL_PATH),
        num_threads=2
    )

    with CameraManager() as camera:
        calibration = load_or_create_calibration(
            pose_detector,
            camera
        )

        print("\nCalibration references:")
        print(f"Neck: {calibration.neck_reference:.1f}")
        print(f"Arm:  {calibration.arm_reference:.1f}")
        print(f"Head: {calibration.head_reference:.1f}")

        print("\nStarting posture analysis. Press Ctrl+C to quit.\n")

        try:
            while True:
                frame = camera.capture_frame()
                landmarks = pose_detector.detect(frame)

                pose = PoseLandmarks(
                    keypoints=landmarks,
                    image_width=frame.shape[1],
                    image_height=frame.shape[0],
                    confidence_threshold=CONFIDENCE_THRESHOLD
                )

                result = analyze_posture(pose, calibration)

                if result.is_valid:
                    print("\nPosture features:")
                    for name, value in result.features.items():
                        print(f"{name:24s}: {value:.3f}")
                else:
                    print(f"\nInvalid frame: {result.reason}")

        except KeyboardInterrupt:
            print("\nPosture analysis stopped.")


if __name__ == "__main__":
    main()
