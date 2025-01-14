import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

calibration_data = {}


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle


def start_calibration():
    global calibration_data
    cap = cv2.VideoCapture(0)
    print("Align yourself. Press 'C' to capture calibration.")

    while True:
        _, frame = cap.read()
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            break

    # calibration measurements
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        _, frame = cap.read()
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = results.pose_landmarks.landmark

        nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        calibration_data['neck_angle'] = calculate_angle(right_shoulder, nose, left_shoulder)
        calibration_data['shoulder_diff'] = abs(left_shoulder[1] - right_shoulder[1])

    cap.release()
    cv2.destroyAllWindows()
    print("Calibration completed.")


def start_monitoring(alert_callback):
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                neck_angle = calculate_angle(right_shoulder, nose, left_shoulder)
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])

                if abs(neck_angle - calibration_data['neck_angle']) > 20:
                    alert_callback("Forward Head Tilt Detected")
                elif abs(shoulder_diff - calibration_data['shoulder_diff']) > 0.03:
                    alert_callback("Shoulder Imbalance Detected")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

