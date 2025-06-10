# 요약된 개선 버전
from ultralytics import YOLO
import streamlit as st
import numpy as np
import cv2
from PIL import Image

model = YOLO("yolov8n-pose.pt")

st.title("YOLOv8 Keypoint Detection")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    results = model.predict(img_np)[0]
    annotated_img = img_np.copy()

    if results.keypoints is not None and len(results.keypoints.xy) > 0:
        for person in results.keypoints.xy:
            for x, y in person.cpu().numpy():
                cv2.circle(annotated_img, (int(x), int(y)), 4, (0, 255, 0), -1)

        st.image(annotated_img, caption="Keypoints Detected", use_column_width=True)
        st.subheader("Keypoint 좌표")
        for i, person in enumerate(results.keypoints.xy):
            st.write(f"Person {i+1}:")
            st.write(np.round(person.cpu().numpy(), 2))
    else:
        st.warning("사람이 감지되지 않았습니다.")
