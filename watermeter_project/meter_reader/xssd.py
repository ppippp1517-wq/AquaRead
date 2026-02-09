import os
import math
import re
import shutil
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
from ultralytics import YOLO
from django.conf import settings

# ตั้งค่าตำแหน่ง tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# โหลดโมเดล YOLO
MODEL_PATH = 'D:/projectCPE/dataset/runs/detect/train12/weights/best.pt'
CNN_MODEL_PATH = 'D:/projectCPE/models/modelneedle.h5'
model = YOLO(MODEL_PATH)

# ฟังก์ชันตรวจจับค่าจากเข็ม
def detect_needle_value(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(" อ่านภาพไม่สำเร็จ:", image_path)
        return None, None

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=30, maxLineGap=10)
    if lines is None:
        print(" ไม่พบเส้น Hough line")
        return None, None

    max_len = 0
    best_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > max_len:
            max_len = length
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        print(" ไม่พบเส้นเข็มที่ดีที่สุด")
        return None, None

    x1, y1, x2, y2 = best_line
    dist1 = np.hypot(x1 - cx, y1 - cy)
    dist2 = np.hypot(x2 - cx, y2 - cy)
    px, py = (x1, y1) if dist1 > dist2 else (x2, y2)

    angle_rad = math.atan2(py - cy, px - cx)
    angle_deg = (math.degrees(angle_rad) + 360) % 360

    # อ่านแบบละเอียด
    value = round((angle_deg / 360) * 10, 1)

    # อ่านแบบแบ่งช่วง
    digit = int(((angle_deg + 18) % 360) // 36)

    return value, digit

def process_meter_image(image_path):
    result = model.predict(source=image_path, conf=0.25)[0]
    df = result.to_df()
    print(df)
    
    if df.empty:
        return {
            'digital_x': '0',
            'x001': '0',
            'x0001': '0',
            'x00001': '0',
            'total': '0.000',
            'detected_image_path': None
        }

    original_img = Image.open(image_path)
    boxes = df['box'].values
    class_ids = df['class'].values
    class_names = df['name'].values

    detection_data = []
    for i in range(len(boxes)):
        detection_data.append({
            'box': boxes[i],
            'class_id': class_ids[i],
            'class_name': class_names[i],
            'confidence': df['confidence'].values[i]
        })

    required_class_ids = [0, 1, 2, 3]
    ocr_result_by_class = {}

    for class_id in required_class_ids:
        candidates = [d for d in detection_data if d['class_id'] == class_id]
        if not candidates:
            ocr_result_by_class[class_id] = '0'
            continue

        best_det = max(candidates, key=lambda x: x['confidence'])
        box = best_det['box']

        x1, y1, x2, y2 = map(int, [box['x1'], box['y1'], box['x2'], box['y2']])
        cropped = original_img.crop((x1, y1, x2, y2))
        enhanced = ImageEnhance.Contrast(cropped).enhance(2.0)

        temp_path = f'temp_crop_class{class_id}.png'
        enhanced.save(temp_path)

        if class_id == 0:
            text = pytesseract.image_to_string(enhanced, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            cleaned_text = re.sub(r'\D', '', text.strip())
            ocr_result_by_class[class_id] = cleaned_text if cleaned_text else '0'
        else:
            value, digit = detect_needle_value(temp_path)
            ocr_result_by_class[class_id] = str(digit) if digit is not None else '0'

        os.remove(temp_path)

    int_part = ocr_result_by_class.get(0, '0')
    decimal1 = ocr_result_by_class.get(1, '0')
    decimal2 = ocr_result_by_class.get(2, '0')
    decimal3 = ocr_result_by_class.get(3, '0')
    combined_number = f"{int_part}.{decimal1}{decimal2}{decimal3}"

    # === เซฟภาพผลลัพธ์ YOLO ที่มีกรอบ ===
    yolo_detect_path = os.path.join('D:/projectCPE/dataset/images/detect_images', f"detected_{os.path.basename(image_path)}")
    plot_img = result.plot()  # ได้ numpy array
    cv2.imwrite(yolo_detect_path, plot_img)

    # === คัดลอกภาพไปยัง media/outputs/ ===
    media_detect_path = os.path.join(settings.MEDIA_ROOT, 'outputs', f"detected_{os.path.basename(image_path)}")
    os.makedirs(os.path.dirname(media_detect_path), exist_ok=True)
    shutil.copy(yolo_detect_path, media_detect_path)

    # Path ที่ใช้ใน HTML
    detected_path = os.path.join('media', 'outputs', f"detected_{os.path.basename(image_path)}")

    return {
        'digital_x': int_part,
        'x001': decimal1,
        'x0001': decimal2,
        'x00001': decimal3,
        'total': combined_number,
        'detected_image_path': f"/media/outputs/detected_{os.path.basename(image_path)}"

    }
