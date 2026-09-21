from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import sqlite3
import time
import pyttsx3
from datetime import datetime
from collections import deque, defaultdict
import requests
import anthropic

model = YOLO("runs/models/robot_hand_yolo-10/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)
class_names = {0: "pen", 1: "cup", 2: "smartphone", 3: "car key", 4: "bottle", 5: "wallet", 6: "glasses",
               7: "hookah tong", 8: "watch", 9: "hair tie", 10: "lighter"}

engine = pyttsx3.init()
engine.setProperty('rate', 160)
ANTHROPIC_KEY = "YOUR_API_KEY_HERE"
ai_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN_HERE"
TELEGRAM_CHAT_ID = "1312518113"

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    except:
        print("  ⚠️ Telegram failed (no internet?)")
PHONE_WIDTH_CM = 7.8
PHONE_HEIGHT_CM = 16.34
DB = "detections.db"
GRIPPER_MARGIN_CM = 1.0   # extra opening beyond object width
GRIPPER_MAX_CM = 8.0      # maximum gripper opening (change to your future arm's spec)

def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_name TEXT, confidence REAL, colors TEXT,
            width_cm REAL, height_cm REAL, text_on_object TEXT,
            grasp_type TEXT, grip_width_cm REAL, grasp_x_cm REAL, grasp_y_cm REAL,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_detection(obj_name, conf, colors_str, w_cm, h_cm, text, grasp_type, grip_w, gx, gy):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO detections (object_name, confidence, colors, width_cm, height_cm, text_on_object, grasp_type, grip_width_cm, grasp_x_cm, grasp_y_cm, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (obj_name, round(conf*100,1), colors_str, w_cm, h_cm, text, grasp_type, grip_w, gx, gy,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    row_id = c.lastrowid
    conn.close()
    return row_id

init_db()

def name_bgr(bgr):
    b, g, r = bgr
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Brightness levels first
    if v < 40: return "Black"
    if v > 220 and s < 30: return "White"

    # Low saturation = gray shades
    if s < 35:
        if v < 100: return "Dark Gray"
        if v < 180: return "Gray"
        return "Light Gray"

    # Chromatic colors (high saturation)
    if v < 80:
        # Dark versions
        if h < 10 or h >= 170: return "Dark Red"
        if h < 22: return "Brown"
        if h < 78: return "Dark Green"
        if h < 130: return "Dark Blue"
        return "Dark Purple"

    # Normal and bright colors
    if h < 5 or h >= 175: return "Red"
    if h < 15: return "Orange"
    if h < 25: return "Gold"
    if h < 33: return "Yellow"
    if h < 45: return "Yellow-Green"
    if h < 78: return "Green"
    if h < 95: return "Teal"
    if h < 110: return "Light Blue"
    if h < 130: return "Blue"
    if h < 145: return "Purple"
    if h < 160: return "Magenta"
    return "Pink"

def get_dominant_colors(roi, k=3, min_percent=15):
    small = cv2.resize(roi, (50, 50))
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    total = counts.sum()
    colors = []
    for i in np.argsort(-counts):
        percent = counts[i] * 100 / total
        if percent >= min_percent:
            cname = name_bgr(centers[i])
            if cname not in [c[0] for c in colors]:
                colors.append((cname, percent))
    return colors

def plan_grasp(obj_name, w_cm, h_cm):
    '''Decide HOW the robot hand should grab this object.
    Returns: (grasp_type, grip_width_cm, approach)'''
    aspect = h_cm / w_cm if w_cm > 0 else 1

    if obj_name in ("pen", "car key", "hookah tong", "watch", "hair tie", "lighter") or w_cm < 3:
        grasp_type = "PINCH"          # precision 2-finger grip
        approach = "top"
    elif aspect > 1.5:
        grasp_type = "SIDE_GRIP"      # tall object: bottle, cup -> grab from side
        approach = "side"
    else:
        grasp_type = "TOP_GRIP"       # flat/wide: phone, wallet -> grab from top
        approach = "top"

    grip_width = min(w_cm + GRIPPER_MARGIN_CM, GRIPPER_MAX_CM)
    graspable = w_cm + GRIPPER_MARGIN_CM <= GRIPPER_MAX_CM
    return grasp_type, round(grip_width, 1), approach, graspable

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
FRAME_W, FRAME_H = 1280, 720

frame_count = 0
cached_boxes = []
cached_text = {}
calib_history = deque(maxlen=15)
size_history = defaultdict(lambda: deque(maxlen=15))
fps_time = time.time()
fps = 0
STABLE_TOLERANCE = 0.08

print("?? Robot Hand AI v3 - Perception + Grasp Planning")
print("  Q=quit | T=text | V=voice | A=ask AI | S=save | G=robot cmd")
print("  Auto-calibration: keep phone visible in frame\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    clean = frame.copy()
    frame_count += 1

    if frame_count % 10 == 0:
        now = time.time()
        fps = 10 / (now - fps_time)
        fps_time = now

    if frame_count % 3 == 0:
        results = model(frame, imgsz=320, verbose=False)
        cached_boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.6:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    obj_name = class_names.get(cls_id, "unknown")
                    roi = clean[y1:y2, x1:x2]
                    colors = get_dominant_colors(roi) if roi.size > 0 else []
                    cached_boxes.append((x1, y1, x2, y2, obj_name, conf, colors))

        for (ax1, ay1, ax2, ay2, aname, aconf, acolors) in cached_boxes:
            if aname == "smartphone" and aconf > 0.75:
                pw, ph = ax2 - ax1, ay2 - ay1
                sample = pw / PHONE_WIDTH_CM if ph >= pw else pw / PHONE_HEIGHT_CM
                calib_history.append(sample)
                break

    pixels_per_cm = float(np.median(calib_history)) if len(calib_history) >= 5 else None

    for (x1, y1, x2, y2, obj_name, conf, colors) in cached_boxes:
        px_w, px_h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2)//2, (y1 + y2)//2   # object center in pixels

        locked = False
        grasp_line = "Grasp: need calibration"
        if pixels_per_cm:
            w_now, h_now = px_w/pixels_per_cm, px_h/pixels_per_cm
            size_history[obj_name].append((w_now, h_now))
            hist = size_history[obj_name]
            ws = [s[0] for s in hist]; hs = [s[1] for s in hist]
            w_med, h_med = float(np.median(ws)), float(np.median(hs))
            if len(hist) >= 10:
                w_var = (max(ws)-min(ws))/w_med if w_med > 0 else 1
                h_var = (max(hs)-min(hs))/h_med if h_med > 0 else 1
                locked = w_var < STABLE_TOLERANCE and h_var < STABLE_TOLERANCE
            size_str = f"{w_med:.1f} x {h_med:.1f} cm" + ("  [LOCKED]" if locked else "  [measuring...]")

            # ----- GRASP PLANNING -----
            gtype, gwidth, approach, graspable = plan_grasp(obj_name, w_med, h_med)
            # grasp point in cm, relative to CAMERA CENTER (0,0 = middle of view)
            gx = round((cx - FRAME_W/2) / pixels_per_cm, 1)
            gy = round((FRAME_H/2 - cy) / pixels_per_cm, 1)
            if graspable:
                grasp_line = f"Grasp: {gtype} open {gwidth}cm @({gx},{gy})"
            else:
                grasp_line = "Grasp: TOO WIDE for gripper!"

            if locked:
                # draw grip points: two red dots where fingers close
                if gtype == "SIDE_GRIP" or gtype == "PINCH":
                    cv2.circle(frame, (x1, cy), 8, (0, 0, 255), -1)
                    cv2.circle(frame, (x2, cy), 8, (0, 0, 255), -1)
                    cv2.line(frame, (x1, cy), (x2, cy), (0, 0, 255), 2)
                else:  # TOP_GRIP
                    cv2.circle(frame, (cx, y1), 8, (0, 0, 255), -1)
                    cv2.circle(frame, (cx, y2), 8, (0, 0, 255), -1)
                    cv2.line(frame, (cx, y1), (cx, y2), (0, 0, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)  # center point
        else:
            size_str = f"{px_w} x {px_h} px (show phone)"

        box_color = (0, 255, 0) if locked else (0, 200, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        color_str = ", ".join([f"{c} {p:.0f}%" for c, p in colors]) or "Unknown"
        text_str = cached_text.get(obj_name, "press T")

        panel_lines = [
            f"{obj_name.upper()}  {conf*100:.0f}%",
            f"Colors: {color_str}",
            f"Size: {size_str}",
            grasp_line,
            f"Text: {text_str}",
        ]
        panel_h = 25 * len(panel_lines) + 10
        panel_y = max(y1 - panel_h, 0)
        panel_w = max(400, 12 * max(len(l) for l in panel_lines))
        cv2.rectangle(frame, (x1, panel_y), (x1 + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        for i, line in enumerate(panel_lines):
            cv2.putText(frame, line, (x1 + 5, panel_y + 22 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    calib_str = f"Calib: {pixels_per_cm:.1f} px/cm" if pixels_per_cm else "Calib: show phone in frame"
    cv2.putText(frame, f"{calib_str} | FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Robot Hand AI v3", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('t'):
        print("📖 Reading text...")
        for (x1, y1, x2, y2, obj_name, conf, colors) in cached_boxes:
            roi = clean[y1:y2, x1:x2]
            if roi.size > 0:
                # Try multiple preprocessing methods, keep best
                candidates = []

                # Method 1: raw color image
                r1 = reader.readtext(roi)
                candidates.append(r1)

                # Method 2: upscaled 2x (helps small text)
                big = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                r2 = reader.readtext(big)
                candidates.append(r2)

                # Method 3: grayscale + mild contrast
                gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
                gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=5)
                r3 = reader.readtext(gray)
                candidates.append(r3)

                # Method 4: adaptive threshold (works on uneven lighting)
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)
                r4 = reader.readtext(thresh)
                candidates.append(r4)

                # Method 5: inverted threshold (white text on dark background)
                inv = cv2.bitwise_not(thresh)
                r5 = reader.readtext(inv)
                candidates.append(r5)

                # Score each method: sum of confidences for words > 0.3
                def score(res):
                    return sum(d[2] for d in res if d[2] > 0.3 and len(d[1]) > 1)

                best = max(candidates, key=score)
                text = ", ".join([d[1] for d in best if d[2] > 0.3 and len(d[1]) > 1]) or "No text"
                cached_text[obj_name] = text
                print(f"  {obj_name}: {text}")
    elif key == ord('g'):
        # Print the robot command for each LOCKED object
        print("\n?? ROBOT COMMANDS:")
        if not pixels_per_cm:
            print("  ? Not calibrated")
        for (x1, y1, x2, y2, obj_name, conf, colors) in cached_boxes:
            hist = size_history[obj_name]
            if pixels_per_cm and len(hist) >= 10:
                ws = [s[0] for s in hist]; hs = [s[1] for s in hist]
                w_med, h_med = float(np.median(ws)), float(np.median(hs))
                gtype, gwidth, approach, graspable = plan_grasp(obj_name, w_med, h_med)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                gx = round((cx - FRAME_W/2)/pixels_per_cm, 1)
                gy = round((FRAME_H/2 - cy)/pixels_per_cm, 1)
                if graspable:
                    print(f"  GRAB(object='{obj_name}', x={gx}, y={gy}, grip_width={gwidth}, type='{gtype}', approach='{approach}')")
                else:
                    print(f"  SKIP('{obj_name}': wider than gripper max {GRIPPER_MAX_CM}cm)")
        print()

    elif key == ord('a'):
        # Build context from current detections
        if not cached_boxes:
            print("❌ No objects detected to ask about")
        else:
            scene = []
            for (x1, y1, x2, y2, obj_name, conf, colors) in cached_boxes:
                color_str = ", ".join([f"{c} {p:.0f}%" for c, p in colors])
                text = cached_text.get(obj_name, "not read yet")
                hist = size_history[obj_name]
                if pixels_per_cm and len(hist) >= 5:
                    ws = [s[0] for s in hist]
                    hs = [s[1] for s in hist]
                    size = f"{float(np.median(ws)):.1f} x {float(np.median(hs)):.1f} cm"
                else:
                    size = "unknown"
                gtype, gwidth, approach, graspable = plan_grasp(obj_name,
                    float(np.median([s[0] for s in hist])) if hist else 5,
                    float(np.median([s[1] for s in hist])) if hist else 5)
                scene.append(f"- {obj_name}: confidence {conf*100:.0f}%, colors: {color_str}, size: {size}, text on it: {text}, grasp: {gtype} open {gwidth}cm")

            scene_text = "\n".join(scene)
            question = input("\n🧠 Ask about the scene: ")

            print("🤖 Thinking...")
            try:
                response = ai_client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=300,
                    system="You are the AI brain of a robotic arm. You see objects through a camera. Answer questions about the detected objects based on the sensor data provided. Be concise, practical, and helpful. If asked to sort or compare, use the real measurements. If asked about danger, use common sense about the objects.",
                    messages=[{"role": "user", "content": f"Objects currently detected by my camera:\n{scene_text}\n\nQuestion: {question}"}]
                )
                answer = response.content[0].text
                print(f"\n🤖 AI: {answer}\n")
                engine.say(answer)
                engine.runAndWait()
            except Exception as e:
                print(f"❌ AI error: {e}")    

    elif key == ord('v'):
        for (x1, y1, x2, y2, obj_name, conf, colors) in cached_boxes:
            color_str = " and ".join([c for c, p in colors])
            text = cached_text.get(obj_name, "no text read yet")
            speech = f"{obj_name}. Color: {color_str}."
            if pixels_per_cm:
                hist = size_history[obj_name]
                if len(hist) >= 5:
                    ws = [s[0] for s in hist]
                    hs = [s[1] for s in hist]
                    speech += f" Size: {float(np.median(ws)):.1f} by {float(np.median(hs)):.1f} centimeters."
            if text != "no text read yet":
                speech += f" Text reads: {text}."
            print(f"🔊 {speech}")
            engine.say(speech)
        engine.runAndWait()   

    elif key == ord('s'):
        if not cached_boxes:
            print("? Nothing to save")
        else:
            print("?? Saving LOCKED detections...")
            saved = 0
            for (x1, y1, x2, y2, obj_name, conf, colors) in cached_boxes:
                hist = size_history[obj_name]
                if pixels_per_cm and len(hist) >= 10:
                    ws = [s[0] for s in hist]; hs = [s[1] for s in hist]
                    w_med = round(float(np.median(ws)), 1)
                    h_med = round(float(np.median(hs)), 1)
                    w_var = (max(ws)-min(ws))/w_med if w_med > 0 else 1
                    if w_var < STABLE_TOLERANCE:
                        gtype, gwidth, approach, graspable = plan_grasp(obj_name, w_med, h_med)
                        cx, cy = (x1+x2)//2, (y1+y2)//2
                        gx = round((cx - FRAME_W/2)/pixels_per_cm, 1)
                        gy = round((FRAME_H/2 - cy)/pixels_per_cm, 1)
                        color_str = ", ".join([f"{c} {p:.0f}%" for c, p in colors]) or "Unknown"
                        text = cached_text.get(obj_name, "Not read")
                        row_id = save_detection(obj_name, conf, color_str, w_med, h_med, text, gtype, gwidth, gx, gy)
                        print(f"  ? #{row_id}: {obj_name} | {w_med}x{h_med}cm | {gtype} {gwidth}cm @({gx},{gy})")
                        send_telegram(f"🤖 Object: {obj_name.upper()}\n🎯 Confidence: {round(conf*100,1)}%\n🎨 Colors: {color_str}\n📏 Size: {w_med} x {h_med} cm\n📝 Text: {text}\n🦾 Grasp: {gtype} open {gwidth}cm\n📍 Position: ({gx}, {gy})")
                        saved += 1
            if saved == 0:
                print("  ? Wait for [LOCKED] before saving")

cap.release()
cv2.destroyAllWindows()
