import cv2
import os

# ---- SETTINGS ----
CLASS_ID = 10        # 9 = hair tie, 10 = lighter  <- CHANGE per session
CLASS_NAME = "lighter"
START_INDEX = 1
TARGET = 40

# Center box (object must fit inside)
BOX_W, BOX_H = 400, 400

os.makedirs("../dataset/images/train", exist_ok=True)
os.makedirs("../dataset/labels/train", exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = START_INDEX
print(f"?? Manual collection: {CLASS_NAME} (class {CLASS_ID})")
print("Hold object INSIDE the box. SPACE = capture, Q = quit\n")

while count <= TARGET:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    x1 = W//2 - BOX_W//2
    y1 = H//2 - BOX_H//2
    x2 = x1 + BOX_W
    y2 = y1 + BOX_H

    display = frame.copy()
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(display, f"{CLASS_NAME}: {count-1}/{TARGET} captured | SPACE=capture Q=quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Manual Collector", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        name = CLASS_NAME.replace(" ", "_")
        img_path = f"../dataset/images/train/{name}_{count}.jpg"
        lbl_path = f"../dataset/labels/train/{name}_{count}.txt"
        cv2.imwrite(img_path, frame)

        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        bw = BOX_W / W
        bh = BOX_H / H
        with open(lbl_path, "w") as f:
            f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"  ? {count}/{TARGET} saved")
        count += 1

cap.release()
cv2.destroyAllWindows()
print("Done!")
