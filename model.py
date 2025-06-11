import cv2
from ultralytics import YOLO

# 모델 로드
model = YOLO("yolov8n-pose.pt")

# 비디오 파일 로드
cap = cv2.VideoCapture("dataset/1.Training/video/fist/C032_A24_SY21_P01_S01_01DAS.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 포즈 추론
    results = model(frame)

    # 키포인트 좌표 출력 (첫 사람만 예시로 출력)
    for kp in results[0].keypoints:
        coords = kp.xy.cpu().numpy()  # (17, 2): 17개 keypoint
        print("Keypoints:", coords)

    # 시각화
    annotated_frame = results[0].plot()

    # 화면에 출력
    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
