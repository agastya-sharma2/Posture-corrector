
import cv2
import numpy as np

from ai_edge_litert.interpreter import Interpreter


class MoveNet:
    KEYPOINT_NAMES = [
        "nose",
        "left_eye", "right_eye",
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ]

    def __init__(self, model_path="models/movenet_lightning.tflite",
                 num_threads=2):

        self.model_path = model_path

        print("Loading MoveNet...")

        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )

        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        input_shape = self.input_details["shape"]

        self.input_height = int(input_shape[1])
        self.input_width = int(input_shape[2])

        self.input_dtype = self.input_details["dtype"]

        print(
            f"MoveNet loaded. Input: "
            f"{self.input_width}x{self.input_height}"
        )

    def _preprocess(self, frame):
        """
        Convert a BGR camera image into the model's input format.

        Preserves aspect ratio using padding.

        Returns:
            input_tensor
            scale
            pad_x
            pad_y
        """

        original_h, original_w = frame.shape[:2]

        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Calculate resize dimensions.
        scale = min(
            self.input_width / original_w,
            self.input_height / original_h
        )

        resized_w = int(round(original_w * scale))
        resized_h = int(round(original_h * scale))

        resized = cv2.resize(
            rgb,
            (resized_w, resized_h)
        )

        # Center the image on a black canvas.
        pad_x = (self.input_width - resized_w) // 2
        pad_y = (self.input_height - resized_h) // 2

        canvas = np.zeros(
            (self.input_height, self.input_width, 3),
            dtype=np.uint8
        )

        canvas[
            pad_y:pad_y + resized_h,
            pad_x:pad_x + resized_w
        ] = resized

        # Convert to the model's expected data type.
        if np.issubdtype(self.input_dtype, np.floating):
            tensor = canvas.astype(self.input_dtype)

        elif self.input_dtype == np.uint8:
            tensor = canvas

        elif self.input_dtype == np.int8:
            scale_factor, zero_point = (
                self.input_details["quantization"]
            )

            if scale_factor == 0:
                raise ValueError("Invalid model quantization.")

            tensor = np.round(
                canvas.astype(np.float32) / scale_factor
                + zero_point
            )

            tensor = np.clip(
                tensor,
                -128,
                127
            ).astype(np.int8)

        else:
            raise TypeError(
                f"Unsupported model input type: {self.input_dtype}"
            )

        input_tensor = np.expand_dims(tensor, axis=0)

        return input_tensor, scale, pad_x, pad_y

    def detect(self, frame):
        """
        Run pose estimation on a BGR image.

        Returns:
            NumPy array of shape (17, 3).

            Each row contains:
                [x, y, confidence]

            Coordinates are in original image pixels.
        """

        if frame is None or frame.size == 0:
            raise ValueError("Invalid input image.")

        original_h, original_w = frame.shape[:2]

        input_tensor, scale, pad_x, pad_y = (
            self._preprocess(frame)
        )

        # Run inference.
        self.interpreter.set_tensor(
            self.input_details["index"],
            input_tensor
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details["index"]
        )

        # MoveNet output: [1, 1, 17, 3]
        keypoints = output.reshape(-1, 3)

        if len(keypoints) != 17:
            raise ValueError(
                f"Expected 17 landmarks, got {len(keypoints)}"
            )

        points = []

        for y, x, confidence in keypoints:

            # Convert normalized model coordinates
            # back into original camera coordinates.

            px = (x * self.input_width - pad_x) / scale
            py = (y * self.input_height - pad_y) / scale

            points.append([
                float(px),
                float(py),
                float(confidence)
            ])

        return np.array(points, dtype=np.float32)
