
from picamera2 import Picamera2
import time


class CameraManager:
    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution
        self.camera = None
        self.started = False

    def start(self):
        """Initialize and start the Raspberry Pi camera."""

        if self.started:
            return

        self.camera = Picamera2()

        config = self.camera.create_video_configuration(
            main={
                "size": self.resolution,
                "format": "BGR888"
            }
        )

        self.camera.configure(config)
        self.camera.start()

        # Allow camera exposure and white balance to settle.
        time.sleep(2)

        self.started = True
        print("Camera initialized.")

    def capture_frame(self):
        """Capture and return a BGR image as a NumPy array."""

        if not self.started:
            raise RuntimeError("Camera has not been started.")

        return self.camera.capture_array()

    def stop(self):
        """Stop and release the camera."""

        if self.camera is not None:
            self.camera.stop()
            self.camera.close()

            self.camera = None
            self.started = False

            print("Camera released.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
