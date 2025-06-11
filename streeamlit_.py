# app.py
import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import pickle

# 🔹 모델 로드
pose_model = YOLO("yolov8n-pose.pt")  # 사전학습된 pose 모델
with open("violence_classifier.pkl", "rb") as f:
    clf_model = pickle.load(f)  # 사전학습된 분류 모델 (MLP, SVC 등)

# 🔹 키포인트 추출 함수
def extract_keypoints_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoint_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model(frame)

        # keypoints가 존재하는지 확인
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            for kp in results[0].keypoints:
                try:
                    coords = kp.xy.cpu().numpy().flatten()
                    if len(coords) == 34:  # 17 keypoints × 2
                        keypoint_list.append(coords)
                except Exception as e:
                    print("⚠️ 예외 발생:", e)
                    continue
        break  # 첫 프레임만 처리 (빠른 데모용)
    cap.release()

    return np.array(keypoint_list)

# 🔹 Streamlit UI
st.title("👁️‍🗨️ 폭행 탐지 서비스 (YOLOv8 + Pose + MLP Classifier)")

uploaded_file = st.file_uploader("🎥 비디오 파일 업로드", type=["mp4", "avi"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        st.video(tmp.name)

        st.write("📌 키포인트 추출 중...")
        keypoints = extract_keypoints_from_video(tmp.name)
        st.write(f"🔎 추출된 키포인트 벡터 수: {keypoints.shape[0]}")

        if keypoints.shape[0] == 0:
            st.error("❌ 키포인트 추출 실패: 사람이 감지되지 않았습니다.")
        else:
            # 첫 프레임 기반 벡터 사용 (또는 여러 프레임 평균도 가능)
            input_vec = keypoints[0]

            st.write("📊 행동 분석 중...")
            prediction = clf_model.predict([input_vec])[0]
            label = "🚨 이상행동 (폭행)" if prediction == 1 else "✅ 정상행동"
            st.subheader(f"📌 분류 결과: {label}")
