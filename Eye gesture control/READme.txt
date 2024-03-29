Real-Time Eye Blink Detection

This project implements real-time eye blink detection using facial landmark detection with dlib and OpenCV. It calculates the eye aspect ratio (EAR) to determine blinks and counts the total number of blinks. Additionally, it incorporates a timer to reset the blink count after a certain interval.

Key Features:

Real-time eye blink detection using dlib and OpenCV
Calculation of the eye aspect ratio (EAR) for blink detection
Counting the total number of blinks
Automatic reset of blink count after a specified interval

Dependencies:

scipy
imutils
dlib
opencv-python

Usage Instructions:

Run the script eye_blink_detection.py.
Specify the path to the facial landmark predictor using the -p or --shape-predictor argument.
Optionally, specify the path to the input video file using the -v or --video argument.
Press the 'q' key to exit the program.