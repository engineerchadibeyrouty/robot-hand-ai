# Robot Hand AI

AI-powered perception system for a robotic arm that detects objects, recognizes colors, measures dimensions, and plans how to grab them.

## What it does
A camera looks at objects on a table and the system identifies what they are, what color they are, how big they are, and tells the robot hand how to grab them.

## Features
- YOLO object detection (11 classes: pen, cup, smartphone, car key, bottle, wallet, glasses, hookah tong, watch, hair tie, lighter)
- Color recognition with 25 HSV-based color names
- Real-size measurement auto-calibrated using a phone in the frame
- OCR text reading for English and Arabic
- Grasp planning (pinch, side grip, top grip)
- Voice output using pyttsx3
- Telegram notifications for saved detections
- AI assistant using Claude API for object Q&A
- SQLite database to store detections

## Tech Stack
- Python
- OpenCV
- YOLOv8 (Ultralytics)
- EasyOCR
- pyttsx3
- Anthropic Claude API
- SQLite
- Raspberry Pi 5 (hardware phase)

## Results
- mAP50: 0.965
- mAP50-95: 0.859
- Trained on 800 images across 11 classes

## Status
Perception system complete — hardware integration with Raspberry Pi coming next
