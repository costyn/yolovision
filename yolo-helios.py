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

# Define function to map keypoints from image space to ILDA space (0x0000 to 0xFFF)
def map_to_ilda_space(keypoints, frame_width, frame_height):
    ilda_points = []
    for x, y in keypoints:
        # Ignore invalid points (0, 0) which aren't detected
        if x == 0 and y == 0:
            continue

        # Normalize x and y to ILDA range
        norm_x = x / frame_width
        norm_y = y / frame_height
        ilda_x = int(norm_x * 0xFFF)
        ilda_y = int((1 - norm_y) * 0xFFF)  # Flip the y-axis for ILDA

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
        keypoints = result.keypoints.cpu().numpy()  # [N, 17, 3] array (x, y, confidence)

        # Map keypoints to ILDA space
        ilda_keypoints = map_to_ilda_space(result.keypoints.xy[0], frame.shape[1], frame.shape[0])  # Use 'xy' for x, y coordinates
        
        # Create a frame of HeliosPoints
        frame_data = (HeliosPoint * len(ilda_keypoints))()
        for i, (x, y) in enumerate(ilda_keypoints):
            frame_data[i] = HeliosPoint(x, y, 255, 255, 255, 255)  # White color, full intensity

        # Send the frame to the Helios DAC
        for device_id in range(numDevices):
            status_attempts = 0
            while status_attempts < 512 and HeliosLib.GetStatus(device_id) != 1:
                status_attempts += 1
            HeliosLib.WriteFrame(device_id, 25000, 0, ctypes.pointer(frame_data), len(ilda_keypoints))

    # Optionally, display the annotated video frame (without bounding boxes)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Pose Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

# Close the Helios DAC connection
HeliosLib.CloseDevices()