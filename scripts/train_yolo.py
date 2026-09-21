from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="../dataset.yaml",
    epochs=15,          # reduced from 40
    imgsz=320,          # reduced from 640 (4x faster)
    batch=8,
    patience=5,
    project="../models",
    name="robot_hand_yolo"
)

print("\n✅ Training done!")