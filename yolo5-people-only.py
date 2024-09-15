import cv2
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)
    # Extract results (bounding boxes, class ids, confidence scores)
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

    # Filter detections to only include people (class id == 0)
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # class ID for 'person' is 0
            # Draw bounding boxes for people
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv5 Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()