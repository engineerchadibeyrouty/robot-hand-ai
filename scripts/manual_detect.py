from ultralytics import YOLO
import cv2
import numpy as np
import easyocr

model = YOLO("runs/models/robot_hand_yolo-9/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)
class_names = {0: "pen", 1: "cup", 2: "smartphone", 3: "car key", 4: "bottle", 5: "wallet", 6: "glasses"}

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

drawing = False
start_x, start_y = 0, 0
frame_copy = None

def get_color_name(bgr):
    b, g, r = bgr
    if r > 200 and g < 100 and b < 100: return "Red"
    elif r < 100 and g > 200 and b < 100: return "Green"
    elif r < 100 and g < 100 and b > 200: return "Blue"
    elif r > 150 and g > 150 and b < 100: return "Yellow"
    else: return "Gray"

def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, frame_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Manual Detection", frame_copy)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(start_x, x), min(start_y, y)
        x2, y2 = max(start_x, x), max(start_y, y)
        
        if x2 - x1 > 20 and y2 - y1 > 20:
            roi = frame[y1:y2, x1:x2]
            
            results = model(roi, verbose=False)
            obj_name = "unknown"
            conf = 0
            
            for result in results:
                for box in result.boxes:
                    if float(box.conf[0]) > conf:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        obj_name = class_names.get(cls_id, "unknown")
            
            avg_color = cv2.mean(roi)[:3]
            color_name = get_color_name(avg_color)
            
            ocr_results = reader.readtext(roi)
            text = ", ".join([d[1] for d in ocr_results if d[2] > 0.5]) or "No text"
            
            print(f"\n{'='*50}")
            print(f"? Object: {obj_name.upper()}")
            print(f"?? Confidence: {conf*100:.1f}%")
            print(f"?? Color: {color_name}")
            print(f"?? Text: {text}")
            print(f"{'='*50}\n")

cv2.namedWindow("Manual Detection")
cv2.setMouseCallback("Manual Detection", mouse_callback)

print("?? Manual Detection Mode")
print("??  Click and drag to draw rectangle around object")
print("?? Release mouse to analyze")
print("Q to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow("Manual Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
