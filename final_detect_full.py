
#ดีเทคและวาดกรอบแค่3กรอบ ของหน้าปัดเข็ม
# -*- coding: utf-8 -*-
import os
import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import pytesseract
from ultralytics import YOLO
import re

# ===================== CONFIG =====================
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

model_path         = r'D:/projectCPE/dataset/runs/detect/train12/weights/best.pt'
test_images_path   = r'D:/projectCPE/dataset/images/test'
detect_images_path = r'D:/projectCPE/dataset/images/detect_images'
output_path        = r'D:/projectCPE/dataset/images/cropped_images'
csv_output_path    = r'D:/projectCPE/dataset/result.csv'

SHOW_CLASS_IDS     = {1, 2, 3}    # วาดกรอบเฉพาะคลาส 1–3
CROP_CLASS_IDS     = {1, 2, 3}    # ครอปเฉพาะคลาส 1–3
RESIZE_WH          = (32, 32)     # ขนาดไฟล์ _resized.png

# สร้างโฟลเดอร์ผลลัพธ์
os.makedirs(detect_images_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

# โหลดโมเดล
model = YOLO(model_path)

# ===================== ฟังก์ชันอ่านค่าเข็ม =====================
def detect_needle_value(image_path):
    """
    อ่าน 'เข็ม' จากรูปครอปด้วย HoughLinesP:
      - หาเส้นที่ยาวที่สุดเป็นเข็ม
      - วัดมุมจากจุดกึ่งกลางภาพ (cx, cy) → angle_deg ∈ [0, 360)
      - value (ละเอียด): สเกล 0–10 (หนึ่งรอบคือ 10)
      - digit (แบ่งช่วง): มุม/36° → 10 ช่อง (0–9)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                            minLineLength=30, maxLineGap=10)
    if lines is None:
        return None, None

    # หาเส้นที่ยาวที่สุด
    max_len = 0
    best_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > max_len:
            max_len = length
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        return None, None

    x1, y1, x2, y2 = best_line

    # ใช้ปลายที่ไกลจุดศูนย์กลางเป็น 'ปลายเข็ม'
    dist1 = np.hypot(x1 - cx, y1 - cy)
    dist2 = np.hypot(x2 - cx, y2 - cy)
    px, py = (x1, y1) if dist1 > dist2 else (x2, y2)

    angle_rad = math.atan2(py - cy, px - cx)
    angle_deg = (math.degrees(angle_rad) + 360) % 360

    # อ่านแบบละเอียด (0–10)
    value = round((angle_deg / 360) * 10, 1)

    # อ่านแบบแบ่งช่วงเป็นหลัก (0–9) ด้วย offset 18°
    digit = int(((angle_deg + 18) % 360) // 36)

    return value, digit

# ===================== RUN PREDICTION =====================
results = model.predict(source=test_images_path, conf=0.25)
all_data = []

for result in results:
    image_path = result.path
    image_name = os.path.basename(image_path)
    print(f"\nกำลังประมวลผล: {image_name}")

    # แปลงผลเป็น DataFrame
    df = result.to_df()
    if df.empty:
        print("ไม่พบวัตถุ")
        # ใส่บรรทัดว่างเพื่อคงรูปแบบ CSV
        all_data.append({
            'image': image_name,
            'digital_x': '0',
            'x001': '0',
            'x0001': '0',
            'x00001': '0',
            'total': '0.000'
        })
        # เซฟภาพต้นฉบับไว้ใน detect_images แบบไม่มีกล่องก็ได้
        raw = cv2.imread(image_path)
        cv2.imwrite(os.path.join(detect_images_path, f"detected_{image_name}"), raw)
        continue

    # ---------- วาดกรอบเฉพาะคลาส 1–3 ----------
    img_vis = cv2.imread(image_path)
    for _, row in df.iterrows():
        cid = int(row["class"])
        if cid not in SHOW_CLASS_IDS:
            continue
        b = row["box"]  # dict: {'x1','y1','x2','y2'}
        x1, y1, x2, y2 = map(int, [b["x1"], b["y1"], b["x2"], b["y2"]])

        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        conf = row["confidence"] if "confidence" in row else None
        label = f'{row["name"]} {conf:.2f}' if conf is not None else str(row["name"])
        cv2.putText(img_vis, label, (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(detect_images_path, f"detected_{image_name}"), img_vis)

    # ---------- เตรียมครอปเฉพาะคลาส 1–3 ----------
    original_img = Image.open(image_path)

    # เก็บ detection ลง list
    detection_data = []
    for _, row in df.iterrows():
        detection_data.append({
            'box': row['box'],                         # {'x1','y1','x2','y2'}
            'class_id': int(row['class']),
            'class_name': row['name']
        })

    # กรองคลาสที่ต้องการ
    filtered = [d for d in detection_data if d['class_id'] in CROP_CLASS_IDS]

    if not filtered:
        print("ไม่มีคลาส 1–3 ให้ครอป")
        all_data.append({
            'image': image_name,
            'digital_x': '0',
            'x001': '0',
            'x0001': '0',
            'x00001': '0',
            'total': '0.000'
        })
        continue

    # ถ้ามีหลายกล่องในคลาสเดียว เลือกพื้นที่มากสุด
    best_by_class = {}
    for det in filtered:
        cid = det['class_id']
        b = det['box']
        x1, y1, x2, y2 = b['x1'], b['y1'], b['x2'], b['y2']
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        if cid not in best_by_class or area > best_by_class[cid]['area']:
            best_by_class[cid] = {'det': det, 'area': area}

    # ตัวแทนแต่ละคลาส (1,2,3)
    filtered_sorted = [best_by_class[cid]['det'] for cid in sorted(best_by_class.keys())]

    # อ่านค่าเข็มจากครอป
    ocr_result_by_class = {}
    for i, det in enumerate(filtered_sorted):
        b = det['box']
        class_id = det['class_id']
        class_name = det['class_name']
        x1, y1, x2, y2 = map(int, [b['x1'], b['y1'], b['x2'], b['y2']])

        cropped = original_img.crop((x1, y1, x2, y2))
        enhanced = ImageEnhance.Contrast(cropped).enhance(2.0)

        crop_path = os.path.join(output_path, f"class{class_id}_{image_name}_{i+1}.png")
        enhanced.save(crop_path)

        resized = enhanced.resize(RESIZE_WH)
        resized.save(crop_path.replace(".png", "_resized.png"))

        value, digit = detect_needle_value(crop_path)
        ocr_result_by_class[class_id] = str(digit) if digit is not None else '0'
        print(f"  🔧 class {class_id} ({class_name}) → angle_value: {value}, mapped_digit: {digit}")

    # ---------- รวมผลเป็นตัวเลข ----------
    int_part = '0'  # ไม่ใช้ class 0 แล้ว → ให้เป็น 0 คงรูปแบบ 0.xyz
    decimal1 = ocr_result_by_class.get(1, '0')
    decimal2 = ocr_result_by_class.get(2, '0')
    decimal3 = ocr_result_by_class.get(3, '0')
    combined_number = f"{int_part}.{decimal1}{decimal2}{decimal3}"

    print("\nค่าที่อ่านได้แต่ละ class:")
    print(f"  x001   (class 1): {decimal1}")
    print(f"  x0001  (class 2): {decimal2}")
    print(f"  x00001 (class 3): {decimal3}")
    print(f"\nผลรวม {image_name}: {combined_number}")

    all_data.append({
        'image': image_name,
        'digital_x': int_part,  # คงคอลัมน์ไว้เพื่อความเข้ากันได้ (กำหนดเป็น 0)
        'x001': decimal1,
        'x0001': decimal2,
        'x00001': decimal3,
        'total': combined_number
    })

# ===================== SAVE CSV =====================
df_csv = pd.DataFrame(all_data)
df_csv.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
print(f"\nบันทึกผลลงไฟล์: {csv_output_path}")
