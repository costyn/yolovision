from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640, device="mps")