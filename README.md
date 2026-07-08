# Posture Corrector

A posture-monitoring application that uses computer vision to detect poor posture while you work. Built with Python, OpenCV, and MediaPipe, this project analyzes your body position and helps you maintain better sitting habits.

## Features

* Real-time posture monitoring using computer vision
* Detects poor posture using body landmarks from MediaPipe
* Two available versions:

  * **v1:** Analyzes a side-view of you sitting at your desk
  * **v2:** Uses your phone camera through DroidCam for live posture monitoring
* Built with:

  * Python
  * OpenCV
  * MediaPipe

---

# Versions

## v1 — Side View Posture Detection

Version 1 analyzes a side-view image or video of you working at a desk.

### Setup

1. Clone the repository:

```bash
git clone https://github.com/agastya-sharma2/Posture-corrector.git
cd Posture-Corrector
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your side-view image/video of yourself working at a desk in the same folder as the program.

4. Run the project:

```bash
python3 PostureCorrector_v1.py
```

---

## v2 — Live Posture Monitoring with DroidCam

Version 2 uses your phone as a wireless camera through the DroidCam application.

### Requirements

* DroidCam installed on your phone
* Phone and computer connected to the same network
* DroidCam IP address and port number

### Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Install and open DroidCam on your phone.

3. Copy the IP address and port provided by the app.

4. Setup your phone so that it sees your wrist, elbow, shoulder, and face from a side view.
 
5. Go into the program and type in your phone's ip into the CAMERA_URL constant

Example:
```python
CAMERA_URL = "http://10.0.0.246:4747/video"
```
  It is a good idea to first navigate to this url in your browser to make sure the camera feed works.

6. Run the project:

```bash
python3 PostureCorrector_v2.py
```

The program will connect to your phone camera and use the live video feed for posture detection.

---

# Project Structure

```text
Posture-Corrector/
│
├── PostureCorrector_v1.py
│
├── PostureCorrector_v2.py
│
├── ClickError.wav
|
├── requirements.txt
│
└── README.md
```
---

# Screenshots / Demonstrations

Coming soon.

---

# Future Improvements

Potential improvements:

* [ ] Add screenshots and demonstrations
* [ ] Improve posture detection accuracy
* [ ] Add more posture metrics
* [ ] Add user configuration options
* [ ] Improve documentation

---

# Contributing

Contributions, suggestions, and improvements are welcome.

A contribution guide will be added in the future.

