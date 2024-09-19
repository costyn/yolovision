import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    results = model(frame, device="mps")

    for result in results:
        result.boxes = None  # This removes the bounding boxes, so only the keypoints will be shown
    
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLOv8 Pose Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()