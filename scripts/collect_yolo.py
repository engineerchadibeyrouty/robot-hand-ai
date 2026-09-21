import cv2
import os
import time
from ultralytics import YOLO

# World model auto-labels the boxes for us
labeler = YOLO("yolov8s-worldv2.pt")

CLASSES = ["pen", "cup", "smartphone", "car key", "bottle", "wallet", "glasses", "hookah tong", "watch", "hair tie", "lighter"]

def collect(class_id, class_name, num_images=40):
    img_dir = os.path.join("..", "dataset", "images", "train")
    lbl_dir = os.path.join("..", "dataset", "labels", "train")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # Tell labeler to look for this object only
    labeler.set_classes([class_name])

    cap = cv2.VideoCapture(0)

    print(f"\n📸 {class_name.upper()}")
    print("Hold the object, press SPACE to start.")
    print("Then SLOWLY rotate it and move it around.")

    started = False
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"{class_name}: {count}/{num_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Collecting", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            started = True
            print("🎬 Capturing... rotate the object slowly")
        elif key == 27:
            break

        if started:
            results = labeler(frame, verbose=False, conf=0.05)[0]

            if len(results.boxes) > 0:
                best = max(results.boxes, key=lambda b: float(b.conf[0]))
                x1, y1, x2, y2 = map(float, best.xyxy[0])

                h, w = frame.shape[:2]
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                ts = int(time.time() * 1000)
                img_path = os.path.join(img_dir, f"{class_name}_{ts}.jpg")
                lbl_path = os.path.join(lbl_dir, f"{class_name}_{ts}.txt")

                cv2.imwrite(img_path, frame)
                with open(lbl_path, "w") as f:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                count += 1
                print(f"✅ {count}/{num_images}")
                time.sleep(0.4)

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Done: {count} labeled images for {class_name}")

if __name__ == "__main__":
    for i, name in enumerate(CLASSES):
        input(f"\n➡️ Get your {name} ready, press ENTER...")
        collect(i, name)

    print("\n✅ ALL DONE! Now run: python train_yolo.py")