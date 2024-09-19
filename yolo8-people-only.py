import cv2
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import collections

# Load the YOLOv8 model (pre-trained on COCO dataset)
model = YOLO('yolov8s.pt')  # You can use 'yolov8n.pt' for a smaller model

# Open the video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

bbox_history = collections.deque(maxlen=3)

def smooth_bbox(new_bbox):
    bbox_history.append(new_bbox)  # Add the new bbox to history
    avg_bbox = [sum(x) / len(x) for x in zip(*bbox_history)]  # Calculate the average of the last few bboxes
    return avg_bbox

def add_margin_top(y1, y2, margin_ratio=0.1):
    """Adds a margin to the top of the bounding box."""
    margin_top = margin_ratio * (y2 - y1)  # Calculate margin as a percentage of height
    adjusted_y1 = y1 - margin_top  # Adjust the top by margin
    return max(adjusted_y1, 0)  # Ensure it doesn't go below 0 (frame boundary)

frame_count = 0
detection_interval = 3  # Run detection every 3 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % detection_interval == 0:
        # Perform object detection with YOLOv8
        results = model(frame)

        # Extract bounding boxes, class IDs, and confidence scores
        detections = results[0].boxes.xyxy.cpu().numpy()  # Extract [x1, y1, x2, y2]
        confs = results[0].boxes.conf.cpu().numpy()  # Extract confidence scores
        classes = results[0].boxes.cls.cpu().numpy()  # Extract class IDs

        # Filter detections to only include people (class id == 0)
        for i, cls in enumerate(classes):
            if int(cls) == 0:  # Class ID for 'person' is 0
                x1, y1, x2, y2 = detections[i]
                conf = confs[i]

                smooth_x1, smooth_y1, smooth_x2, smooth_y2 = smooth_bbox([x1, y1, x2, y2])
                adjusted_y1 = add_margin_top(smooth_y1, smooth_y2, margin_ratio=0.1)
                cv2.rectangle(frame, (int(smooth_x1), int(adjusted_y1)), (int(smooth_x2), int(smooth_y2)), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv8 Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()