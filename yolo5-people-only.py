import cv2
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import collections

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

bbox_history = collections.deque(maxlen=3) 

def ease_with_history(current_bbox, easing_factor=0.5):
    """Eases between the previous and current bounding boxes using history."""
    if len(bbox_history) < 2:
        # Not enough history yet, just return the current bounding box
        return current_bbox

    # Get the most recent (previous) bounding box from history
    prev_bbox = bbox_history[0]
    
    # Apply easing between the previous and current bounding boxes
    eased_bbox = [
        prev_value + (curr_value - prev_value) * (easing_factor ** 2)
        for prev_value, curr_value in zip(prev_bbox, current_bbox)
    ]
    
    return eased_bbox

def add_margin_top(y1, y2, margin_ratio=0.1):
    """Adds a margin to the top of the bounding box."""
    margin_top = margin_ratio * (y2 - y1)  # Calculate margin as a percentage of height
    adjusted_y1 = y1 - margin_top  # Adjust the top by margin
    return max(adjusted_y1, 0)  # Ensure it doesn't go below 0 (frame boundary)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    # Extract results (bounding boxes, class ids, confidence scores)
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

    # Filter detections to only include people (class id == 0)
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) == 0:  # class ID for 'person' is 0
            # Current bounding box
            current_bbox = [x1, y1, x2, y2]

            # Smooth bounding boxes using easing
            eased_bbox = ease_with_history(current_bbox)

            # Update the history with the current bounding box
            bbox_history.append(current_bbox)

            # Unpack eased bounding box
            eased_x1, eased_y1, eased_x2, eased_y2 = eased_bbox

            # Add margin to the top of the bounding box
            adjusted_y1 = add_margin_top(eased_y1, eased_y2, margin_ratio=0.1)

            # Draw the adjusted bounding box
            cv2.rectangle(frame, (int(eased_x1), int(adjusted_y1)), (int(eased_x2), int(eased_y2)), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('YOLOv5 Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()