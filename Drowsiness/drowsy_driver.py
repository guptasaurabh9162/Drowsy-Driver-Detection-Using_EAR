import cv2
import pygame
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp

pygame.mixer.init()
sound = pygame.mixer.Sound('MV27TES-alarm.mp3')

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 100

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture('http://192.168.65.207:4747/video')

frame_counter = 0
drowsy = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            leftEye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            rightEye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
            
            leftEye = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) for p in leftEye]
            rightEye = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) for p in rightEye]
            
            cv2.polylines(frame, [np.array(leftEye, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.polylines(frame, [np.array(rightEye, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
            
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            ear = (leftEAR + rightEAR) / 2.0
            
            if ear < EYE_AR_THRESH:
                frame_counter += 1
                if frame_counter >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not drowsy:
                        sound.play()
                        drowsy = True
            else:
                frame_counter = 0
                if drowsy:
                    sound.stop()
                    drowsy = False
    
    cv2.imshow('Drowsiness Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()