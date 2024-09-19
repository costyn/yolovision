import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (pre-trained on COCO dataset)
model = YOLO('yolov8s.pt')

# Open the video capture (0 for default webcam)
capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    results = model(frame,device="mps")

    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLOv8 Person Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()