Real-Time Drowsiness Detection

This project performs real-time drowsiness detection using facial landmark detection with dlib and OpenCV. It calculates the eye aspect ratio (EAR) to detect eye closure indicative of drowsiness. When drowsiness is detected, it triggers an alert to notify the user.

Key Features:

Real-time drowsiness detection using dlib and OpenCV
Calculation of the eye aspect ratio (EAR) for drowsiness detection
Triggering an alert when drowsiness is detected

Dependencies:

scipy
imutils
dlib
opencv-python
pyttsx3

Usage Instructions:

Run the script detect_drowsiness.py.
Specify the path to the facial landmark predictor using the -p argument.
The system will start detecting drowsiness in the video stream from your camera.
If drowsiness is detected, an alert will be triggered.
