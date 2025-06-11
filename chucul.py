import os
import numpy as np
import cv2
from ultralytics import YOLO

pose_model = YOLO('yolov8n-pose.pt')

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints_all = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model(frame)
        for kp in results[0].keypoints:
            if kp.xy.shape[1] == 2:
                keypoints = kp.xy.cpu().numpy().flatten()
                keypoints_all.append(keypoints)
        break
    cap.release()
    return keypoints_all

X = []
y = []


video_label_pairs = [
    (0, 'dataset/1.Training/video/bat/C031_A23_SY21_P01_S06_10DAS.mp4'),     
    (1, 'cut_videos/bat/C031_A23_SY21_P01_S11_08DAS_cut.mp4'),               
]

for label, video_path in video_label_pairs:
    print(f"{video_path}")
    kps = extract_keypoints(video_path)
    for kp in kps:
        if len(kp) == 34:
            X.append(kp)
            y.append(label)

X = np.array(X)
y = np.array(y)

# 저장
np.save("X_keypoints.npy", X)
np.save("y_labels.npy", y)

print(f"X.shape = {X.shape}, y.shape = {y.shape}")
