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

keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

confidence_threshold = 0.5


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
            confidences = result.keypoints.conf[0].cpu().numpy()  # Confidence scores

            for i, name in enumerate(keypoint_names):
                if confidences[i] > confidence_threshold:  # Only send if confidence is high
                    x, y = keypoints[i][0].item(), keypoints[i][1].item()

                    # Send each keypoint x and y as individual OSC messages
                    client.send_message(f"/pose/{name}/x", x)
                    client.send_message(f"/pose/{name}/y", y)

            print("Sent keypoints for detected person.")

    # Optionally, display the annotated video frame
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Pose Detection (Keypoints)', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and destroy windows
capture.release()
cv2.destroyAllWindows()