from flask import Flask, render_template_string, request
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
model = YOLO("runs/models/robot_hand_yolo-9/weights/best.pt")
reader = easyocr.Reader(['en'], gpu=False)
class_names = {0: "pen", 1: "cup", 2: "smartphone", 3: "car key", 4: "bottle", 5: "wallet", 6: "glasses"}

current_image = None

def get_color_name(bgr):
    b, g, r = bgr
    if r > 200 and g < 100 and b < 100: return "Red"
    elif r < 100 and g > 200 and b < 100: return "Green"
    elif r < 100 and g < 100 and b > 200: return "Blue"
    elif r > 150 and g > 150 and b < 100: return "Yellow"
    else: return "Gray"

@app.route('/')
def index():
    html = '''<!DOCTYPE html><html><head><title>Robot Hand AI - Manual Detection</title><style>body{font-family:Arial;background:#111;color:#fff;text-align:center}.container{max-width:1000px;margin:0 auto;padding:20px}h1{color:#0f0}#canvas{border:2px solid #0f0;max-width:800px;cursor:crosshair}#info{background:#222;padding:15px;margin-top:20px;text-align:left;border-radius:5px}.button-group{margin:20px 0}button{padding:10px 20px;margin:5px;background:#0f0;color:#000;border:none;border-radius:5px;cursor:pointer;font-weight:bold}button:hover{background:#0a0}</style></head><body><div class="container"><h1>?? Robot Hand AI - Manual Detection</h1><input type="file" id="upload" accept="image/*"><br><button onclick="capture()">?? Capture from Webcam</button><canvas id="canvas"></canvas><p style="color:#888">Click and drag to draw rectangle, then click to analyze</p><div id="info"><h3>?? Analysis:</h3><div id="result">Upload image or capture from webcam first</div></div></div><script>let canvas=document.getElementById('canvas');let ctx=canvas.getContext('2d');let img=null;let isDrawing=false;let startX,startY;document.getElementById('upload').addEventListener('change',e=>{let reader=new FileReader();reader.onload=f=>{img=new Image();img.onload=()=>{canvas.width=img.width;canvas.height=img.height;ctx.drawImage(img,0,0)};img.src=f.target.result};reader.readAsDataURL(e.target.files[0])});canvas.addEventListener('mousedown',e=>{if(!img)return;isDrawing=true;startX=e.offsetX;startY=e.offsetY});canvas.addEventListener('mousemove',e=>{if(!img||!isDrawing)return;ctx.drawImage(img,0,0);ctx.strokeStyle='#0f0';ctx.lineWidth=2;ctx.rect(startX,startY,e.offsetX-startX,e.offsetY-startY);ctx.stroke()});canvas.addEventListener('mouseup',e=>{if(!img||!isDrawing)return;isDrawing=false;let x1=Math.min(startX,e.offsetX),y1=Math.min(startY,e.offsetY),x2=Math.max(startX,e.offsetX),y2=Math.max(startY,e.offsetY);analyze(x1,y1,x2,y2)});function capture(){let v=document.createElement('video');let c=document.createElement('canvas');navigator.mediaDevices.getUserMedia({video:true}).then(stream=>{v.srcObject=stream;v.play();setTimeout(()=>{c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);img=new Image();img.src=c.toDataURL();img.onload=()=>{canvas.width=img.width;canvas.height=img.height;ctx.drawImage(img,0,0)};stream.getTracks().forEach(t=>t.stop())},1000)}).catch(e=>alert('Webcam error'))}function analyze(x1,y1,x2,y2){if(!img)return;let c=document.createElement('canvas');c.width=x2-x1;c.height=y2-y1;let imgData=ctx.getImageData(x1,y1,x2-x1,y2-y1);c.getContext('2d').putImageData(imgData,0,0);fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:c.toDataURL()})}).then(r=>r.json()).then(d=>{document.getElementById('result').innerHTML='<b>'+d.object+'</b><br>Confidence: '+d.confidence+'<br>Color: '+d.color+'<br>Text: '+d.text})}</script></body></html>'''
    return render_template_string(html)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    results = model(frame, verbose=False)
    
    best_obj = "unknown"
    best_conf = 0
    color = "gray"
    text = "No text"
    
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                cls_id = int(box.cls[0])
                best_obj = class_names.get(cls_id, "unknown")
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                avg_color = cv2.mean(roi)[:3]
                color = get_color_name(avg_color)
                
                ocr_results = reader.readtext(roi)
                text = ", ".join([d[1] for d in ocr_results if d[2] > 0.5]) or "No text"
    
    return {
        "object": best_obj.upper(),
        "confidence": f"{best_conf*100:.1f}%",
        "color": color,
        "text": text
    }

if __name__ == '__main__':
    print("?? Server running at http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
