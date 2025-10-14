from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from collections import deque, Counter
import re

model_path = r"D:\yolo_car_plate\runs\detect\yolo_car_plate10\weights\best.pt"
model = YOLO(model_path)

reader = easyocr.Reader(['en'])

def read_plate(plate_img):
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(plate_gray)
    text = ""
    for (bbox, txt, prob) in results:
        if prob > 0.4:
            text += txt
    digits = re.findall(r'\d', text)
    if len(digits) == 7:
        return text.strip()
    return ""

input_path = r"D:\yolo_car_plate\input.mp4"
output_path = r"D:\yolo_car_plate\output.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
panel_width = 300
out = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width + panel_width, height))

N = 10  
plate_history = deque(maxlen=N)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    current_texts = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            plate_crop = frame[y1:y2, x1:x2]
            text = read_plate(plate_crop)
            if text:
                current_texts.append(text)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    if current_texts:
        plate_history.extend(current_texts)

    if plate_history:
        most_common_text = Counter(plate_history).most_common(1)[0][0]
    else:
        most_common_text = ""

    panel = np.full((height, panel_width, 3), 50, dtype=np.uint8)
    if most_common_text:
        cv2.putText(panel, most_common_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    combined = np.hstack((frame, panel))
    out.write(combined)

cap.release()
out.release()
