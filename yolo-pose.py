import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
            result.boxes = None
            keypoints = result.keypoints.cpu().numpy()  # Get keypoints for each person
            print(keypoints)  # Inspect keypoints (shape will be [N, 17, 3] for N persons)
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow('YOLOv8 Person Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


cv2.imshow("YOLO Results", results[-1].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()