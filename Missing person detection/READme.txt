Real-Time Face Detection and Image Capture

This project implements real-time face detection using OpenCV's Haar cascade classifier and allows users to capture images of detected faces. The captured images can be saved for further processing, such as face recognition.

Key Features:

Real-time face detection using OpenCV
Image capture of detected faces
Saving captured face images for future use

Dependencies:

imutils
argparse
OpenCV (cv2)
os

Usage Instructions:

Run the script face_detection_and_capture.py.
Specify the path to the face cascade XML file using the -c or --cascade argument.
Specify the directory where captured face images will be saved using the -o or --output argument.
Press the 'k' key to capture and save images of detected faces.
Press the 'q' key to exit the program.