import cv2
from ultralytics import YOLO
import ctypes
import numpy as np

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

# Use a constant number of points for each frame
FRAME_POINTS = 1000

# Initialize the YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')

# Open the video capture (0 for default webcam)
capture = cv2.VideoCapture(0)

# Define function to map normalized keypoints to ILDA space (0x0000 to 0xFFF)
def map_to_ilda_space(keypoints):
    ilda_points = []
    for x, y in keypoints:
        # Convert normalized x and y to ILDA range
        ilda_x = int(x * 0xFFF)
        ilda_y = int((1 - y) * 0xFFF)  # Flip the y-axis for ILDA
        ilda_points.append((ilda_x, ilda_y))
    
    return ilda_points

# Main loop: Capture video, process, and send to DAC
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Perform object detection using YOLOv8 pose model
    results = model(frame, device="mps")

    # Iterate over results and extract keypoints (focusing on all keypoints)
    for result in results:
        result.boxes = None
        keypoints = result.keypoints.cpu().numpy()  # [N, 17, 3] array (x, y, confidence)

        keypoints = result.keypoints.xyn[0]  # Normalized x, y coordinates

        # Map keypoints to ILDA space
        ilda_keypoints = map_to_ilda_space(keypoints)

        # Create a frame of HeliosPoints (just the keypoints, no lines)
        frame_points = []
        for x, y in ilda_keypoints:
            frame_points.append(HeliosPoint(x, y, 255, 255, 255, 255))  # White color, full intensity

        # Fill remaining points with black to match FRAME_POINTS
        while len(frame_points) < FRAME_POINTS:
            frame_points.append(HeliosPoint(0, 0, 0, 0, 0, 0))

        # Convert to the correct frame type
        helios_frame = (HeliosPoint * FRAME_POINTS)(*frame_points[:FRAME_POINTS])

        # Send the frame to the Helios DAC
        for device_index in range(numDevices):
            status_attempts = 0
            while status_attempts < 512 and HeliosLib.GetStatus(device_index) != 1:
                status_attempts += 1

            HeliosLib.WriteFrame(device_index, 25000, 0, ctypes.pointer(helios_frame), FRAME_POINTS)

    # Optionally, display the annotated video frame (without bounding boxes)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Pose Detection (Keypoints)', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and DAC resources
capture.release()
HeliosLib.CloseDevices()
cv2.destroyAllWindows()