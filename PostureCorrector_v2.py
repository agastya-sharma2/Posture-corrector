import cv2
import mediapipe as mp
import math
import os
import time
from playsound3 import playsound


title = r"""

 _______  ___   _______    __   __  _______    ______    ___   _______  __   __  _______    __   __  _______ 
|       ||   | |       |  |  | |  ||       |  |    _ |  |   | |       ||  | |  ||       |  |  | |  ||       |
|  _____||   | |_     _|  |  | |  ||    _  |  |   | ||  |   | |    ___||  |_|  ||_     _|  |  |_|  ||____   |
| |_____ |   |   |   |    |  |_|  ||   |_| |  |   |_||_ |   | |   | __ |       |  |   |    |       | ____|  |
|_____  ||   |   |   |    |       ||    ___|  |    __  ||   | |   ||  ||       |  |   |    |       || ______|
 _____| ||   |   |   |    |       ||   |      |   |  | ||   | |   |_| ||   _   |  |   |     |     | | |_____ 
|_______||___|   |___|    |_______||___|      |___|  |_||___| |_______||__| |__|  |___|      |___|  |_______|

"""

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"
BOLD = "\033[1m"

CAMERA_URL = "[REPLACE WITH YOUR PHONE'S IP ADDRESS]"
THREASHOLD = 8

#Angle Function
def angle(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1] #Dot formula is just multiplying x coords and multiplying y coords and adding them
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2) #Length of 1st vector
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2) #Length of 2nd vector (both use pythagoream theorem)

    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    return math.degrees(math.acos(cos_angle)) #Formula for angle

# ---------SETUP---------
def BuildModel(path):
    global fontScale
    global fontWeight
    global fontColor

    global e_coords
    global s_coords
    global eb_coords
    global w_coords
    global er_coords
    global n_coords

    global image

    #Loading Image and setup
    path = path
    image = cv2.imread(path)

    #Loading and Creating pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)

    # Convert to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    #Just Checking for Somebody in the Photo
    if not results.pose_landmarks:
        #print("No person detected")
        raise TypeError("No Person Detected in the frame!")

    h, w, _ = image.shape
    lm = results.pose_landmarks.landmark #Landmarks

    fontScale = 5
    fontWeight = 10
    fontColor = (255, 255, 255) #BGR

    #Gathering Stats for Desired Landmarks
    eye = lm[mp_pose.PoseLandmark.RIGHT_EYE]
    shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
    wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
    ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]
    nose = lm[mp_pose.PoseLandmark.NOSE]

    #Coords of Landmarks
    e_coords = (eye.x * w, eye.y * h)
    s_coords = (shoulder.x * w, shoulder.y * h)
    eb_coords = (elbow.x * w, elbow.y * h)
    w_coords = (wrist.x * w, wrist.y * h)
    er_coords = (ear.x * w, ear.y * h)
    n_coords = (nose.x * w, nose.y * h)

# ---------MATH---------
def PerformMath():
    NeckAngle = 0
    ArmAngle = 0
    HeadAngle = 0

    #Neck
    NeckVector = (e_coords[0] - s_coords[0], e_coords[1] - s_coords[1]) #Think of this as slope
    VerticalVector = (0, -1) #A straight vertical line
    NeckAngle = round(angle(NeckVector, VerticalVector), 1)
    NeckAngle = min(NeckAngle, 180 - NeckAngle) #Just finding the smaller angle
    print("\n Angle of neck: " + str(NeckAngle))

    #Arm
    ArmVector = (eb_coords[0] - w_coords[0], eb_coords[1] - w_coords[1]) #Think of this as slope
    ShoulderVector = (s_coords[0] - eb_coords[0], s_coords[1], eb_coords[1])
    ArmAngle = round(angle(ShoulderVector, ArmVector), 1)
    ArmAngle = max(ArmAngle, 180 - ArmAngle) #Just finding the smaller angle
    print("\n Angle of arm: " + str(ArmAngle))

    #Head
    HeadVector = (n_coords[0] - er_coords[0], n_coords[1] - er_coords[1]) #Think of this as slope
    HorizontalVector = (-1, 0) #A straight horizontal line
    HeadAngle = round(angle(HeadVector, HorizontalVector), 0)
    HeadAngle = min(HeadAngle, 180 - HeadAngle) #Just finding the smaller angle
    print("\n Angle of head: " + str(HeadAngle))

    Plot(image, ArmAngle, NeckAngle, HeadAngle)

    return ArmAngle, NeckAngle, HeadAngle

# ---------PLOTTING---------
def Plot(image, ArmAngle, NeckAngle, HeadAngle):
    #Plotting Points
    for pt in (e_coords, s_coords, eb_coords, w_coords, er_coords, n_coords):
        cv2.circle(image, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)

    #Neck
    cv2.line(image,
        (int(s_coords[0]), int(s_coords[1])), #Shoulder
        (int(e_coords[0]), int(e_coords[1])), #Eye
        (255, 0, 0), 3)

    vertical_length = e_coords[1]  # pixel length for display
    vertical_end = (int(s_coords[0]), int(s_coords[1] - vertical_length))

    cv2.line(image,
        (int(s_coords[0]), int(s_coords[1])), #Shoulder
        vertical_end, #Vertical Line
        (255, 0, 0), 3)

    cv2.putText(image,
        f"{NeckAngle:.1f} degrees",
        (int(s_coords[0] + 60), int(s_coords[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontWeight)

    #Arms
    cv2.line(image,
        (int(w_coords[0]), int(w_coords[1])), #Wrist
        (int(eb_coords[0]), int(eb_coords[1])), #Elbow
        (0, 0, 255), 3)

    cv2.line(image,
        (int(s_coords[0]), int(s_coords[1])),
        (int(eb_coords[0]), int(eb_coords[1])),
        (0, 0, 255), 3)


    cv2.putText(image,
        f"{ArmAngle:.1f} degrees",
        (int(eb_coords[0] + 60), int(eb_coords[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontWeight)

    #Head
    cv2.line(image,
        (int(n_coords[0]), int(n_coords[1])), #Wrist
        (int(er_coords[0]), int(er_coords[1])), #Elbow
        (0, 255, 255), 3)

    cv2.putText(image,
        f"{HeadAngle:.1f} degrees",
        (int(n_coords[0] + 60), int(n_coords[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, fontScale, fontColor, fontWeight)
    

# ---------ANALYZATION---------
def Analyze(ArmAngle, NeckAngle, HeadAngle):
    print(f"\n---------{BLUE}Analyzation{RESET}---------")
    #Score is defind by how many degrees off you are
    NeckScore = 0
    ArmScore = 0
    HeadScore = 0

    #Neck
    NeckScore = abs(RNeckAngle-NeckAngle)
    print("Neck:", str(round(NeckScore, 2)) + "° off")

    #Arm
    ArmScore = abs(RArmAngle-ArmAngle)
    print("Arm:", str(round(ArmScore, 2)) + "° off")

    #Head
    HeadScore = abs(RHeadAngle-HeadAngle)
    print("Head:", str(round(NeckScore, 2)) + "° off")

    print("")

    #Threashold
    if NeckScore > THREASHOLD:
        print("!! Neck angle is not ok!")
        playsound("ClickError.wav")
    if ArmScore > THREASHOLD:
        print("!! Arm angle is not ok!")
        playsound("ClickError.wav")
    if HeadScore > THREASHOLD:
        print("!! Head angle is not ok!")
        playsound("ClickError.wav")
    if NeckScore < THREASHOLD or ArmScore < THREASHOLD or HeadScore < THREASHOLD:
        print("All good!")
    
# ---------CALIBRATION---------
def Calibrate():
    global RArmAngle, RNeckAngle, RHeadAngle

    print(f"---------{BLUE}Calibration Mode{RESET}---------")
    print("Please sit in desired position for the photo. Press Enter when ready")
    input("")

    userSatisfied = 0

    while userSatisfied == 0:

        print("Taking Picture...")
        time.sleep(2)

        cap = cv2.VideoCapture(CAMERA_URL)

        if not cap.isOpened():
            print("!! Could not connect to camera")
            quit()
        else:
            time.sleep(1)

            for _ in range(10):
                cap.read()

            ret, frame = cap.read()

            if ret:
                cv2.imwrite("CalibratePosture.jpg", frame)
                BuildModel("CalibratePosture.jpg")
                RArmAngle, RNeckAngle, RHeadAngle = PerformMath()
        
        cap.release()

        print("Photo taken! Please check if you're satisfied with it!")
        userSatisfied = int(input("1 --> Satisfied, 0 --> Not Satisfied  "))

    print("Succesfully Calibrated!")


# ---------POSTURE MODE---------
def PostureMode():
    print(f"\n---------{BLUE}Posture Mode{RESET}---------")
    print("Please sit in the desired position")

    time.sleep(60)

    cap = cv2.VideoCapture(CAMERA_URL)

    if not cap.isOpened():
        print("!! Could not connect to camera")
        quit()
    else:
        time.sleep(1)

        for _ in range(10):
            cap.read()

        ret, frame = cap.read()

        if ret:
            cv2.imwrite("GivenPosture.jpg", frame)
            BuildModel("GivenPosture.jpg")
            Analyze(*PerformMath())
    
    cap.release()


# ---------START---------
os.system("clear")
print(CYAN,title,RESET)
time.sleep(2)
Calibrate()

# ---------MAIN LOOP---------
option = 0
while 1:
    print(f"\n---------{BLUE}Main Menu{RESET}---------")
    print("What would you like to do?")
    print("1 -> Check Posture")
    print("2 -> Calibrate")
    print("3 -> Quit")
    option = int(input("-> "))
    if option == 3:
        print("Bye!")
        quit()
    elif option == 1:
        while True:
            try:
                PostureMode()
            except KeyboardInterrupt():
                break
    elif option == 2:
        Calibrate()




