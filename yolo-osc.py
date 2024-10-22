import cv2
from ultralytics import YOLO
from pythonosc import udp_client

# Initialize the YOLOv8 pose model
model = YOLO('yolov8n-pose.pt')
client_ip = "localhost"
client_port = 8010

# Open the video capture (0 for default webcam)
capture = cv2.VideoCapture(0)

client = udp_client.SimpleUDPClient(client_ip, client_port)

# Main loop: Capture video, process, and print keypoints
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Perform object detection using YOLOv8 pose model
    results = model(frame, device="mps")

    # Iterate over results and extract keypoints
    for result in results:
        result.boxes = None

        # Check if there are any keypoints detected
        if result.keypoints.has_visible:
            keypoints = result.keypoints.xyn[0]  # Normalized x, y coordinates
            # Extract only the "head" keypoints (nose, left eye, right eye)
            # head_keypoints = keypoints[[0, 1, 2]].cpu().numpy()  # [Nose, Left Eye, Right Eye]

            # print("Head keypoints (Nose, Left Eye, Right Eye):", head_keypoints)

            nose = keypoints[0]  # Extract nose data (index 0)

            nose_x, nose_y = nose[0].item(), nose[1].item()  # Extract x and y as floats

            # Send the nose X and Y coordinates via OSC
            client.send_message("/nose/x", nose_x)
            client.send_message("/nose/y", nose_y)
        else:
            print("No keypoints detected in this frame")

    # Optionally, display the annotated video frame
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Pose Detection (Keypoints)', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and destroy windows
capture.release()
cv2.destroyAllWindows()