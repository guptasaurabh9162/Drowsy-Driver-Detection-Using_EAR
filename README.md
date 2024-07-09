# Drowsy-Driver-Detection-Using_EAR
This project implements a real-time drowsiness detection system to enhance driver safety. Utilizing the Eye Aspect Ratio (EAR) method, it identifies when a driver's eyes are closed, indicating potential drowsiness, and provides an alert. The system employs OpenCV for image processing and Haar cascades for face and eye detection.
Features
- Real-time video capture from webcam
- Face and eye detection using Haar cascades
- Calculation of Eye Aspect Ratio (EAR)
- Visual alert for drowsiness detection

### Technologies Used
- Python
- OpenCV
- NumPy
- SciPy

### How It Works
1. Captures video frames using a webcam.
2. Detects faces and eyes in the frames.
3. Computes the Eye Aspect Ratio (EAR) for each eye.
4. Displays a visual alert if EAR indicates eyes are closed.

### Usage
1. Connect a driod webcam in your mobile,Change the IP adress in the code
2. Install required libraries
3.  opencv-python, numpy, scipy.
4. Run the script: python drowsiness_detection.py.
