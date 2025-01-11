import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    
    return angle

def detect_posture():
    cap = cv2.VideoCapture(0)

    # Posture variables
    forward_head_tilt = False
    shoulder_imbalance = False
    sideward_bend = False

    # Counters for sustained bad posture
    fht_counter = 0
    si_counter = 0
    sb_counter = 0
    
    # Define thresholds
    FHT_THRESHOLD = 150
    SI_THRESHOLD = 0.03
    SB_THRESHOLD = 15

    print("Please sit in your normal posture for calibration. Press 'c' when ready.")
    while True:
        _, frame = cap.read()
        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    # Calibration measurements
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        _, frame = cap.read()
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = results.pose_landmarks.landmark
        
        # Get initial measurements
        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        
        calibration_neck_angle = calculate_angle(right_shoulder, nose, left_shoulder)
        calibration_shoulder_level = abs(left_shoulder[1] - right_shoulder[1])
        calibration_spine_angle = calculate_angle(nose, [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2], 
                                              [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2])

    # Alert variables
    alert_start_time = 0
    alert_duration = 5
    current_alert = ""

    # Main detection loop
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get current measurements
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

                neck_angle = calculate_angle(right_shoulder, nose, left_shoulder)
                shoulder_level = abs(left_shoulder[1] - right_shoulder[1])
                spine_angle = calculate_angle(nose, [(left_hip[0] + right_hip[0])/2, (left_hip[1] + right_hip[1])/2], 
                                         [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2])

                bad_posture_message = ""

                # Detect posture issues
                if abs(neck_angle - calibration_neck_angle) > FHT_THRESHOLD:
                    cv2.putText(image, 'Forward Head Tilt', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    bad_posture_message += "Forward Head Tilt detected. "

                if abs(shoulder_level - calibration_shoulder_level) > SI_THRESHOLD:
                    cv2.putText(image, 'Shoulder Imbalance', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    bad_posture_message += "Shoulder Imbalance detected. "

                if abs(spine_angle - calibration_spine_angle) > SB_THRESHOLD:
                    cv2.putText(image, 'Sideward Bend', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    bad_posture_message += "Sideward Bend detected. "

                # Display measurements
                cv2.putText(image, f'Neck angle: {neck_angle:.2f}', (10,120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(image, f'Shoulder level: {shoulder_level:.4f}', (10,150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.putText(image, f'Spine angle: {spine_angle:.2f}', (10,180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            except:
                pass

            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Posture Detection', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_posture()
