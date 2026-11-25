
from camera.camera_manager import CameraManager
from vision.movenet import MoveNet

import cv2
from pathlib import Path


MODEL_PATH = Path(__file__).parent / "models" / "movenet_lightning.tflite"


def main():

    # Initialize MoveNet.
    pose_detector = MoveNet(
        model_path=str(MODEL_PATH),
        num_threads=2
    )

    # Capture an image and run inference.
    with CameraManager() as camera:

        print("Capturing image...")

        frame = camera.capture_frame()

        print("Running pose detection...")

        landmarks = pose_detector.detect(frame)

        print("\nDetected landmarks:\n")

        for name, (x, y, confidence) in zip(
            MoveNet.KEYPOINT_NAMES,
            landmarks
        ):
            print(
                f"{name:16s} "
                f"x={x:7.1f} "
                f"y={y:7.1f} "
                f"confidence={confidence:.3f}"
            )

        # Save captured image for inspection.
        output_path = Path(__file__).parent / "capture.jpg"

        cv2.imwrite(str(output_path), frame)

        print(f"\nImage saved: {output_path}")


if __name__ == "__main__":
    main()
