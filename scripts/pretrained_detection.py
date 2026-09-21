import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO

model = YOLO("runs/models/robot_hand_yolo-10/weights/best.pt")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

age_net = cv2.dnn.readNet('age_net.caffemodel', 'age_deploy.prototxt')
AGE_RANGES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']

def get_color_name(image, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    cx1 = x1 + int(w * 0.3)
    cx2 = x2 - int(w * 0.3)
    cy1 = y1 + int(h * 0.3)
    cy2 = y2 - int(h * 0.3)
    roi = image[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return "Unknown"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    color_mask = (s_ch > 60) & (v_ch > 60)
    color_ratio = np.sum(color_mask) / (roi.shape[0] * roi.shape[1])
    if color_ratio < 0.20:
        avg_v = np.mean(v_ch)
        if avg_v > 180: return "White"
        elif avg_v > 120: return "Silver/Gray"
        elif avg_v > 60: return "Dark Gray"
        else: return "Black"
    hue = np.median(h_ch[color_mask])
    sat = np.median(s_ch[color_mask])
    val = np.median(v_ch[color_mask])
    if hue < 8 or hue > 172: name = "Red"
    elif hue < 20: name = "Orange"
    elif hue < 33: name = "Yellow"
    elif hue < 78: name = "Green"
    elif hue < 98: name = "Cyan"
    elif hue < 128: name = "Blue"
    elif hue < 150: name = "Purple"
    else: name = "Pink"
    if val > 200 and sat < 120: name = "Light " + name
    elif val < 90: name = "Dark " + name
    return name

def detect_age(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = frame[y:y+h, x:x+w]
    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227),
                                  (78.4263377603, 87.7689143744, 114.895847746),
                                  swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_RANGES[age_preds[0].argmax()]
    return age

def analyze(cap, num_frames=15):
    names, colors, confs, sizes, ages = [], [], [], [], []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        results = model(frame, verbose=False, conf=0.4)[0]
        best = None
        for box in results.boxes:
            conf = float(box.conf[0])
            if best is None or conf > best[0]:
                best = (conf, box)
        if best:
            conf, box = best
            class_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            names.append(class_name)
            confs.append(conf)
            colors.append(get_color_name(frame, x1, y1, x2, y2))
            sizes.append((x2 - x1, y2 - y1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        age = detect_age(frame)
        if age:
            ages.append(age)
        cv2.imshow("Robot Hand AI", frame)
        cv2.waitKey(1)
    if not names:
        return None, None
    final_name = Counter(names).most_common(1)[0][0]
    final_color = Counter(colors).most_common(1)[0][0]
    best_conf = max(c for n, c in zip(names, confs) if n == final_name) * 100
    consistency = names.count(final_name) / len(names) * 100
    avg_w = int(np.mean([s[0] for s in sizes]))
    avg_h = int(np.mean([s[1] for s in sizes]))
    final_age = Counter(ages).most_common(1)[0][0] if ages else None
    return (final_name, best_conf, consistency, final_color, avg_w, avg_h), final_age

def main():
    cap = cv2.VideoCapture(0)
    print("\n" + "=" * 50)
    print("🤖 ROBOT HAND AI - YOUR CUSTOM MODEL")
    print("=" * 50)
    print("Hold object steady → press SPACE to analyze")
    print("Press Q to quit")
    print("=" * 50)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, "SPACE = analyze | Q = quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Robot Hand AI", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:
            print("\n🔍 Analyzing...")
            result, age = analyze(cap)
            print("=" * 50)
            if result:
                name, conf, consistency, color, w, h = result
                print(f"✅ OBJECT      : {name}")
                print(f"📊 CONFIDENCE  : {conf:.1f}%")
                print(f"🔁 CONSISTENCY : {consistency:.0f}%")
                print(f"🎨 COLOR       : {color}")
                print(f"📐 DIMENSIONS  : {w} x {h} px")
            else:
                print("❌ No object detected")
            if age:
                print(f"👤 AGE         : {age}")
            else:
                print("👤 AGE         : No face detected")
            print("=" * 50)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()