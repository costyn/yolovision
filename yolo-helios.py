import cv2
from ultralytics import YOLO
import ctypes

# Define HeliosPoint structure
class HeliosPoint(ctypes.Structure):
    _fields_ = [('x', ctypes.c_uint16),
                ('y', ctypes.c_uint16),
                ('r', ctypes.c_uint8),
                ('g', ctypes.c_uint8),
                ('b', ctypes.c_uint8),
                ('i', ctypes.c_uint8)]

# Load and initialize Helios DAC library
HeliosLib = ctypes.cdll.LoadLibrary("./libHeliosDacAPI.dylib")
numDevices = HeliosLib.OpenDevices()
print("Found ", numDevices, "Helios DACs")

# Initialize the YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Open the video capture (0 for default webcam)
capture = cv2.VideoCapture(0)

# Common pose connections (17 keypoints) – this is just an example, based on COCO dataset order
connections = [
    (0, 1), (1, 2),  # Nose -> Left eye -> Left ear
    (0, 3), (3, 4),  # Nose -> Right eye -> Right ear
    (5, 6),          # Left shoulder -> Right shoulder
    (5, 7), (7, 9),  # Left shoulder -> Left elbow -> Left wrist
    (6, 8), (8, 10), # Right shoulder -> Right elbow -> Right wrist
    (11, 12),        # Left hip -> Right hip
    (11, 13), (13, 15), # Left hip -> Left knee -> Left ankle
    (12, 14), (14, 16)  # Right hip -> Right knee -> Right ankle
]

# Define function to map keypoints from image space to ILDA space (0x0000 to 0xFFF)
def map_to_ilda_space(keypoints):
    ilda_points = []
    for x, y in keypoints:
        ilda_x = int(x * 0xFFF)
        ilda_y = int((1 - y) * 0xFFF)  # Flip the y-axis for ILDA

        ilda_points.append((ilda_x, ilda_y))
    
    return ilda_points

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Perform object detection using YOLOv8 pose model
    results = model(frame, device="mps")

    # Iterate over results and extract keypoints
    for result in results:
        result.boxes = None
        keypoints = result.keypoints.cpu().numpy()  # [N, 17, 3] array (x, y, confidence)

        # Map keypoints to ILDA space
        ilda_keypoints = map_to_ilda_space(result.keypoints.xyn[0])  # Use 'xyn' for normalized coordinates
        
        ilda_points = []
        for start, end in connections:
            if start < len(ilda_keypoints) and end < len(ilda_keypoints):
                start_point = ilda_keypoints[start]
                end_point = ilda_keypoints[end]
                
                if start_point != (0, 0) and end_point != (0, 0):  # Skip invalid points
                    # Append both start and end points to create a line between them
                    ilda_points.append(HeliosPoint(start_point[0], start_point[1], 255, 255, 255, 255))
                    ilda_points.append(HeliosPoint(end_point[0], end_point[1], 255, 255, 255, 255))

        # Send the frame to the Helios DAC
        frame_data = (HeliosPoint * len(ilda_points))(*ilda_points)  # Convert to HeliosPoint array
        for device_id in range(numDevices):
            status_attempts = 0
            while status_attempts < 512 and HeliosLib.GetStatus(device_id) != 1:
                status_attempts += 1
            HeliosLib.WriteFrame(device_id, 25000, 0, ctypes.pointer(frame_data), len(ilda_points))

    # Optionally, display the annotated video frame (without bounding boxes)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Pose Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

# Close the Helios DAC connection
HeliosLib.CloseDevices()