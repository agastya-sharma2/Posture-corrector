import cv2
import mediapipe as mp
import math
import os
import time
import sys

from playsound3 import playsound


# ==================== CONFIGURATION ====================

TITLE = r"""
 _______  ___   _______    __   __  _______    ______    ___   _______  __   __  _______    __   __  _______ 
|       ||   | |       |  |  | |  ||       |  |    _ |  |   | |       ||  | |  ||       |  |  | |  ||       |
|  _____||   | |_     _|  |  | |  ||    _  |  |   | ||  |   | |    ___||  |_|  ||_     _|  |  |_|  ||____   |
| |_____ |   |   |   |    |  |_|  ||   |_| |  |   |_||_ |   | |   | __ |       |  |   |    |       | ____|  |
|_____  ||   |   |   |    |       ||    ___|  |    __  ||   | |   ||  ||       |  |   |    |       ||____|  |
 _____| ||   |   |   |    |       ||   |      |   |  | ||   | |   |_| ||   _   |  |   |     |     |  ____|  |
|_______||___|   |___|    |_______||___|      |___|  |_||___| |_______||__| |__|  |___|      |___|  |_______|

"""

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"

CAMERA_URL = "http://10.0.0.245:4747/video"

THRESHOLD = 8

IMAGE_CALIBRATION = "CalibratePosture.jpg"
IMAGE_POSTURE = "GivenPosture.jpg"

FONT_SCALE = 1.2
FONT_WEIGHT = 2
FONT_COLOR = (255, 255, 255)

mp_pose = mp.solutions.pose


# ==================== ANGLE CALCULATIONS ====================

def calculate_angle(v1, v2):
    """Calculate the angle between two 2D vectors."""

    dot = v1[0] * v2[0] + v1[1] * v2[1]

    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))


# ==================== CAMERA ====================

def capture_image(filename):
    """Capture an image from the camera and save it."""

    print("Taking picture...")

    cap = cv2.VideoCapture(CAMERA_URL)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not connect to camera!")

    try:
        time.sleep(1)

        # Allow camera frames to stabilize.
        for _ in range(10):
            ret, frame = cap.read()

            if not ret:
                raise RuntimeError("Failed to read camera frame!")

        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Failed to capture image!")

        if not cv2.imwrite(filename, frame):
            raise RuntimeError("Failed to save image!")

    finally:
        cap.release()

    print("Picture saved!")


# ==================== POSE DETECTION ====================

def build_model(path):
    """Detect pose landmarks from an image."""

    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    height, width, _ = image.shape

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1
    ) as pose:

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

    if not results.pose_landmarks:
        raise ValueError("No person detected in the frame!")

    landmarks = results.pose_landmarks.landmark

    # Store relevant landmarks.
    landmark_names = {
        "eye": mp_pose.PoseLandmark.RIGHT_EYE,
        "shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
        "wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
        "ear": mp_pose.PoseLandmark.RIGHT_EAR,
        "nose": mp_pose.PoseLandmark.NOSE
    }

    coords = {}

    for name, landmark_id in landmark_names.items():

        landmark = landmarks[landmark_id]

        coords[name] = (
            landmark.x * width,
            landmark.y * height
        )

    return image, coords


# ==================== POSTURE MATH ====================

def perform_math(image, coords):

    # Neck
    neck_vector = (
        coords["eye"][0] - coords["shoulder"][0],
        coords["eye"][1] - coords["shoulder"][1]
    )

    vertical_vector = (0, -1)

    neck_angle = calculate_angle(neck_vector, vertical_vector)
    neck_angle = min(neck_angle, 180 - neck_angle)

    # Arm
    arm_vector = (
        coords["elbow"][0] - coords["wrist"][0],
        coords["elbow"][1] - coords["wrist"][1]
    )

    shoulder_vector = (
        coords["shoulder"][0] - coords["elbow"][0],
        coords["shoulder"][1] - coords["elbow"][1]
    )

    arm_angle = calculate_angle(shoulder_vector, arm_vector)

    # Preserve original larger-angle convention.
    arm_angle = max(arm_angle, 180 - arm_angle)

    # Head
    head_vector = (
        coords["nose"][0] - coords["ear"][0],
        coords["nose"][1] - coords["ear"][1]
    )

    horizontal_vector = (-1, 0)

    head_angle = calculate_angle(head_vector, horizontal_vector)
    head_angle = min(head_angle, 180 - head_angle)

    neck_angle = round(neck_angle, 1)
    arm_angle = round(arm_angle, 1)
    head_angle = round(head_angle, 1)

    print(f"\nNeck angle: {neck_angle} degrees")
    print(f"Arm angle: {arm_angle} degrees")
    print(f"Head angle: {head_angle} degrees")

    plot(image, coords, arm_angle, neck_angle, head_angle)

    return arm_angle, neck_angle, head_angle


# ==================== VISUALIZATION ====================

def plot(image, coords, arm_angle, neck_angle, head_angle):

    def point(name):
        return tuple(map(int, coords[name]))

    # Draw landmarks.
    for coordinate in coords.values():
        cv2.circle(
            image,
            tuple(map(int, coordinate)),
            6,
            (0, 255, 0),
            -1
        )

    # Neck.
    cv2.line(image, point("shoulder"), point("eye"), (255, 0, 0), 3)

    cv2.line(
        image,
        point("shoulder"),
        (
            point("shoulder")[0],
            point("shoulder")[1] - 100
        ),
        (255, 0, 0),
        3
    )

    cv2.putText(
        image,
        f"{neck_angle:.1f} degrees",
        (point("shoulder")[0] + 30, point("shoulder")[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        FONT_COLOR,
        FONT_WEIGHT
    )

    # Arm.
    cv2.line(image, point("wrist"), point("elbow"), (0, 0, 255), 3)
    cv2.line(image, point("shoulder"), point("elbow"), (0, 0, 255), 3)

    cv2.putText(
        image,
        f"{arm_angle:.1f} degrees",
        (point("elbow")[0] + 30, point("elbow")[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        FONT_COLOR,
        FONT_WEIGHT
    )

    # Head.
    cv2.line(image, point("nose"), point("ear"), (0, 255, 255), 3)

    cv2.putText(
        image,
        f"{head_angle:.1f} degrees",
        (point("nose")[0] + 30, point("nose")[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        FONT_COLOR,
        FONT_WEIGHT
    )

    # Display image.
    cv2.imshow("Posture Analysis", image)
    cv2.waitKey(1)


# ==================== ANALYSIS ====================

def analyze(arm_angle, neck_angle, head_angle):

    print(f"\n---------{BLUE}Analysis{RESET}---------")

    scores = {
        "Neck": abs(REFERENCE_NECK - neck_angle),
        "Arm": abs(REFERENCE_ARM - arm_angle),
        "Head": abs(REFERENCE_HEAD - head_angle)
    }

    errors_found = False

    for body_part, score in scores.items():

        print(f"{body_part}: {score:.2f} degrees off")

        if score > THRESHOLD:
            print(f"{RED}!! {body_part} angle is not okay!{RESET}")
            errors_found = True

    if errors_found:
        try:
            playsound("ClickError.wav")
        except Exception as error:
            print(f"Could not play sound: {error}")

    else:
        print(f"{GREEN}All good!{RESET}")


# ==================== CALIBRATION ====================

REFERENCE_ARM = None
REFERENCE_NECK = None
REFERENCE_HEAD = None


def calibrate():

    global REFERENCE_ARM
    global REFERENCE_NECK
    global REFERENCE_HEAD

    print(f"\n---------{BLUE}Calibration Mode{RESET}---------")

    input("Sit in your desired position and press Enter when ready.")

    while True:

        try:
            capture_image(IMAGE_CALIBRATION)

            image, coords = build_model(IMAGE_CALIBRATION)

            arm, neck, head = perform_math(image, coords)

        except (RuntimeError, ValueError, FileNotFoundError) as error:
            print(f"{RED}Error: {error}{RESET}")
            continue

        print("\nPlease check your calibration image.")

        choice = input("1 --> Satisfied, 0 --> Retry: ").strip()

        if choice == "1":

            REFERENCE_ARM = arm
            REFERENCE_NECK = neck
            REFERENCE_HEAD = head

            print(f"\n{GREEN}Successfully calibrated!{RESET}")
            return

        elif choice == "0":
            print("Retrying calibration...")

        else:
            print("Invalid choice. Enter 0 or 1.")


# ==================== POSTURE MODE ====================

def posture_mode():

    print(f"\n---------{BLUE}Posture Mode{RESET}---------")
    print("Please sit in your desired position.")

    time.sleep(10)

    try:
        capture_image(IMAGE_POSTURE)

        image, coords = build_model(IMAGE_POSTURE)

        angles = perform_math(image, coords)

        analyze(*angles)

    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"{RED}Error: {error}{RESET}")

    except Exception as error:
        print(f"{RED}Unexpected error: {error}{RESET}")


# ==================== MAIN MENU ====================

def main():

    os.system("cls" if os.name == "nt" else "clear")

    print(f"{CYAN}{TITLE}{RESET}")

    time.sleep(2)

    calibrate()

    while True:

        print(f"\n---------{BLUE}Main Menu{RESET}---------")
        print("1 -> Check Posture")
        print("2 -> Calibrate")
        print("3 -> Quit")

        option = input("-> ").strip()

        if option == "1":

            while True:

                try:
                    posture_mode()

                    again = input(
                        "\nPress Enter to check again or type 'q' to return: "
                    ).strip().lower()

                    if again == "q":
                        break

                except KeyboardInterrupt:
                    print("\nReturning to menu...")
                    break

        elif option == "2":
            calibrate()

        elif option == "3":
            print("Bye!")

            cv2.destroyAllWindows()
            sys.exit(0)

        else:
            print("Invalid option. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nProgram interrupted.")

    finally:
        cv2.destroyAllWindows()
