#Se ejecutara segundo

import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic

def extract_keypoints(image, results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def process_images(data_dir):
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for seq in os.listdir(class_dir):
            seq_dir = os.path.join(class_dir, seq)
            if not os.path.isdir(seq_dir):
                continue
            for frame in os.listdir(seq_dir):
                if frame.endswith('.jpg'):
                    image_path = os.path.join(seq_dir, frame)
                    image = cv2.imread(image_path)
                    with mp_holistic.Holistic(static_image_mode=True) as holistic:
                        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                        keypoints = extract_keypoints(image, results)
                        np.save(os.path.join(seq_dir, frame.replace('.jpg', '.npy')), keypoints)

if __name__ == "__main__":
    data_dir = 'sign_language_dataset'
    process_images(data_dir)
